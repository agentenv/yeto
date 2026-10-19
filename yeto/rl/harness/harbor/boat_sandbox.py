"""One task container inside one boat VM (S25 N10.8); no Harbor import.

TB2 tasks name a prebuilt image (``[environment].docker_image``) and expect
to run inside it.  A boat sandbox is a VM with Docker, so this class:

1. creates the VM (quota slot + start, idempotent create),
2. ``docker run -d`` the task image as container ``main`` (``sleep infinity``),
3. runs every exec as ``docker exec`` through boat's detached command API
   (polled; the synchronous API caps a command at 600 s),
4. moves files through the VM: signed upload/download URL to a staging path
   under ``/tmp``, then ``docker cp``; directories travel as one tar archive
   (modes, symlinks and empty directories survive; a truncated archive fails),
5. deletes the VM in :meth:`close`.  :meth:`close` is idempotent and also runs
   when :meth:`start` fails, so a failed or cancelled trial leaves no VM.

The boat client is synchronous; calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import tarfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .boat_client import FAILED_STATES, READY_STATES, BoatClient, BoatError, BoatQuota

logger = logging.getLogger(__name__)

CONTAINER = "main"
STAGING_DIR = "/tmp/yeto-xfer"  # the boat file API only reaches /home/user and /tmp
DOCKER = "sudo -n docker"

# Machine types by guaranteed memory (GiB), read with `boat run --type small` on
# 2026-10-19: small = 2 vCPU / 3 GiB.  Larger types are taken for tasks that ask
# for more than small holds; their sizes come from the API at create time.
SMALL_MEMORY_MB = 3 * 1024
DEFAULT_TTL_S = 3 * 3600


def machine_type_for(memory_mb: int | None, cpus: int | None) -> str:
    """Smallest boat type that fits the task (small is billed at half rate)."""
    if (memory_mb or 0) <= SMALL_MEMORY_MB - 512 and (cpus or 1) <= 2:
        return "small"
    if (memory_mb or 0) <= 6 * 1024 and (cpus or 1) <= 4:
        return "default"
    return "large"


@dataclass(frozen=True)
class ContainerSpec:
    image: str
    workdir: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ExecOutput:
    stdout: str
    stderr: str
    return_code: int


class BoatSandbox:
    def __init__(self, client: BoatClient, quota: BoatQuota, *, name: str,
                 machine_type: str = "small", ttl_s: int = DEFAULT_TTL_S,
                 ready_timeout_s: float = 300.0, pull_timeout_s: float = 1800.0,
                 poll_interval_s: float = 1.0, sleep: Callable[[float], object] = asyncio.sleep) -> None:
        self._client = client
        self._quota = quota
        self._name = name
        self._machine_type = machine_type
        self._ttl_s = ttl_s
        self._ready_timeout_s = ready_timeout_s
        self._pull_timeout_s = pull_timeout_s
        self._poll_s = poll_interval_s
        self._sleep = sleep
        self._holds_slot = False
        self.sandbox_id: str | None = None
        self.usage_seconds: float | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self, spec: ContainerSpec) -> None:
        if self._holds_slot:
            raise BoatError("boat sandbox already started")
        await self._quota.acquire()
        self._holds_slot = True
        try:
            self.sandbox_id = await asyncio.to_thread(
                self._client.create, machine_type=self._machine_type, ttl_seconds=self._ttl_s,
                idempotency_key=f"yeto-{self._name}-{uuid.uuid4().hex}")
            logger.info("boat sandbox %s created for %s (%s)", self.sandbox_id, self._name,
                        self._machine_type)
            await self._wait_ready()
            await self._run_container(spec)
        except BaseException:
            await self.close()
            raise

    async def _wait_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ready_timeout_s
        while True:
            state = await asyncio.to_thread(self._client.state, self.sandbox_id)
            if state in READY_STATES:
                return
            if state in FAILED_STATES:
                raise BoatError(f"boat sandbox {self.sandbox_id} entered state {state}")
            if loop.time() > deadline:
                raise BoatError(f"boat sandbox {self.sandbox_id} not ready after "
                                f"{self._ready_timeout_s:.0f} s (state {state})")
            await self._sleep(2.0)

    async def _run_container(self, spec: ContainerSpec) -> None:
        image = shlex.quote(spec.image)
        # Bridge network as Harbor's docker provider: with --network host the task's own
        # servers (e.g. sshd on :22) would collide with the VM's.
        run = [DOCKER, "run", "-d", "--name", CONTAINER, "--add-host", "host.docker.internal:host-gateway"]
        if spec.cpus:
            run += ["--cpus", str(spec.cpus)]
        if spec.memory_mb:
            run += ["--memory", f"{spec.memory_mb}m"]
        if spec.workdir:
            run += ["-w", shlex.quote(spec.workdir)]
        for key, value in (spec.env or {}).items():
            run += ["-e", shlex.quote(f"{key}={value}")]
        run += [image, "sh", "-c", shlex.quote("sleep infinity")]
        command = f"{DOCKER} pull -q {image} >/dev/null && {' '.join(run)}"
        result = await self.vm_exec(command, timeout_s=self._pull_timeout_s)
        if result.return_code != 0:
            raise BoatError(f"docker run {spec.image} failed ({result.return_code}): "
                            f"{(result.stderr or result.stdout)[-2000:]}")
        if spec.workdir:
            await self.exec(f"mkdir -p {shlex.quote(spec.workdir)}", user="root", timeout_s=60)

    async def close(self) -> None:
        """Delete the VM (idempotent).  Errors are logged: the caller is often unwinding."""
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        try:
            if sandbox_id is not None:
                try:
                    self.usage_seconds = await asyncio.to_thread(self._client.usage_seconds, sandbox_id)
                except Exception:  # usage is a metric; never block the delete
                    logger.warning("boat usage read failed for %s", sandbox_id)
                await asyncio.shield(asyncio.to_thread(self._client.delete, sandbox_id))
                logger.info("boat sandbox %s deleted (%s billable s)", sandbox_id, self.usage_seconds)
        except Exception:
            # ttlSeconds still stops it; `boat list` shows it for manual cleanup.
            logger.exception("boat sandbox %s delete failed; delete it by hand", sandbox_id)
        finally:
            if self._holds_slot:
                self._holds_slot = False
                self._quota.release()

    # -- exec ----------------------------------------------------------------

    async def vm_exec(self, command: str, *, timeout_s: float | None) -> ExecOutput:
        """Run ``command`` on the VM; past ``timeout_s`` it is stopped and RuntimeError raised."""
        sandbox_id = self._require_started()
        process_id = await asyncio.to_thread(self._client.start_command, sandbox_id, command)
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
            logger.warning("boat stop_command %s/%s failed", sandbox_id, process_id)

    async def exec(self, command: str, *, cwd: str | None = None, env: Mapping[str, str] | None = None,
                   timeout_s: float | None = None, user: str | int | None = None) -> ExecOutput:
        """Run ``command`` with bash in the task container (as ``docker exec``)."""
        return await self.vm_exec(docker_exec_command(command, cwd=cwd, env=env, user=user),
                                  timeout_s=timeout_s)

    # -- files ---------------------------------------------------------------

    def _staging(self) -> str:
        return f"{STAGING_DIR}/{uuid.uuid4().hex}"

    async def _checked(self, command: str, what: str, timeout_s: float = 600.0) -> None:
        result = await self.vm_exec(command, timeout_s=timeout_s)
        if result.return_code != 0:
            raise BoatError(f"{what} failed ({result.return_code}): {(result.stderr or result.stdout)[-2000:]}")

    async def upload_bytes(self, data: bytes, target_path: str) -> None:
        sandbox_id = self._require_started()
        staged = self._staging()
        await asyncio.to_thread(self._client.put_file, sandbox_id, staged, data)
        target = shlex.quote(target_path)
        parent = shlex.quote(str(Path(target_path).parent))
        await self._checked(
            f"{DOCKER} exec -u root {CONTAINER} mkdir -p {parent} && "
            f"{DOCKER} cp {shlex.quote(staged)} {CONTAINER}:{target}; rc=$?; rm -f {shlex.quote(staged)}; exit $rc",
            f"upload to {target_path}")

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        sandbox_id = self._require_started()
        archive = await asyncio.to_thread(pack_dir, source_dir)
        staged = self._staging()
        await asyncio.to_thread(self._client.put_file, sandbox_id, f"{staged}.tar.gz", archive)
        s, t = shlex.quote(staged), shlex.quote(target_dir)
        await self._checked(
            f"mkdir -p {s} && tar -xzf {s}.tar.gz -C {s} && "
            f"{DOCKER} exec -u root {CONTAINER} mkdir -p {t} && "
            f"{DOCKER} cp {s}/. {CONTAINER}:{t}; rc=$?; rm -rf {s} {s}.tar.gz; exit $rc",
            f"upload dir to {target_dir}")

    async def download_bytes(self, source_path: str) -> bytes:
        sandbox_id = self._require_started()
        staged = self._staging()
        s = shlex.quote(staged)
        await self._checked(f"mkdir -p {STAGING_DIR} && {DOCKER} cp -L {CONTAINER}:{shlex.quote(source_path)} {s} "
                            f"&& sudo -n chmod a+r {s}", f"download {source_path}")
        try:
            return await asyncio.to_thread(self._client.get_file, sandbox_id, staged)
        finally:
            await self.vm_exec(f"sudo -n rm -f {s}", timeout_s=60)

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        sandbox_id = self._require_started()
        staged = self._staging()
        s = shlex.quote(staged)
        await self._checked(
            f"mkdir -p {s} && {DOCKER} cp {CONTAINER}:{shlex.quote(source_dir.rstrip('/') + '/.')} {s} && "
            f"sudo -n tar -czf {s}.tar.gz -C {s} . && sudo -n chmod a+r {s}.tar.gz",
            f"download dir {source_dir}")
        try:
            archive = await asyncio.to_thread(self._client.get_file, sandbox_id, f"{staged}.tar.gz")
        finally:
            await self.vm_exec(f"sudo -n rm -rf {s} {s}.tar.gz", timeout_s=60)
        await asyncio.to_thread(extract_dir, archive, target_dir)

    def _require_started(self) -> str:
        if self.sandbox_id is None:
            raise BoatError("boat sandbox is not started (or already deleted)")
        return self.sandbox_id


def docker_exec_command(command: str, *, cwd: str | None, env: Mapping[str, str] | None,
                        user: str | int | None) -> str:
    """``docker exec`` line for the VM shell; every argument is shell-quoted.

    Env values travel on the command line, so boat's command history (kept
    90 days) holds them: pass no secrets through ``env``.
    """
    parts = [DOCKER, "exec"]
    if cwd:
        parts += ["-w", shlex.quote(cwd)]
    for key, value in (env or {}).items():
        parts += ["-e", shlex.quote(f"{key}={value}")]
    if user is not None:
        parts += ["-u", shlex.quote(str(user))]
    parts += [CONTAINER, "bash", "-c", shlex.quote(command)]
    return " ".join(parts)


def pack_dir(source_dir: Path | str) -> bytes:
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory {source} does not exist")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(source, arcname=".")
    return buffer.getvalue()


def extract_dir(archive: bytes, target_dir: Path | str) -> None:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        tar.extractall(target, filter="data")  # no absolute paths, no escapes

