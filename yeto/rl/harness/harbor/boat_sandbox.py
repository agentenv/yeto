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

S26 E8: :attr:`BoatSandbox.timings` records how long each start phase took
(``create_s``, ``ready_s``, ``pull_s``, ``run_s``), and a started VM can be
reused for the next task (:meth:`BoatSandbox.recycle` then
:meth:`BoatSandbox.reuse`, driven by :class:`~.boat_pool.BoatSandboxPool`).
With a :class:`CodexCache`, the pinned Codex build is fetched once per VM
(checksum-checked, in parallel with the image pull), mounted read-only into
the task container and put on its PATH as ``codex``; Harbor's codex agent
then finds the pinned version and skips its own install (apt + nvm + npm).

The boat client is synchronous; calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import shlex
import time
import tarfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .boat_client import FAILED_STATES, READY_STATES, BoatClient, BoatError, BoatQuota

logger = logging.getLogger(__name__)
T = TypeVar("T")

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
class CodexCache:
    """A Codex build plus a Node build kept on the VM, mounted at :data:`CODEX_MOUNT`.

    Layout: the Codex npm package at the root, the wrapper at ``bin/codex``,
    Node (official linux-x64 tarball) under ``node/``.
    """

    version: str
    url: str
    sha512: str
    node_version: str
    node_url: str
    node_sha256: str

    @property
    def host_dir(self) -> str:
        return f"{CODEX_CACHE_ROOT}/{self.version}-node{self.node_version}"

    def fetch_command(self) -> str:
        """VM shell line: fetch, check and unpack the build once (atomic rename); exit 0 when ready."""
        d = shlex.quote(self.host_dir)
        musl = f"{CODEX_MOUNT}/vendor/x86_64-unknown-linux-musl"
        wrapper = (f"#!/bin/sh\nPATH=\"{musl}/codex-path:$PATH\"\nexport PATH\n"
                   f"exec {musl}/bin/codex \"$@\"\n")
        return (
            f"[ \"$(uname -m)\" = x86_64 ] && {{ [ -x {d}/bin/codex ] || {{ "
            f"t=$(mktemp -d) && curl -fsSL --retry 3 -o $t/c.tgz {shlex.quote(self.url)} && "
            f"echo {shlex.quote(self.sha512)}\"  $t/c.tgz\" | sha512sum -c --quiet && "
            f"curl -fsSL --retry 3 -o $t/n.tgz {shlex.quote(self.node_url)} && "
            f"echo {shlex.quote(self.node_sha256)}\"  $t/n.tgz\" | sha256sum -c --quiet && "
            f"sudo -n mkdir -p $t/x/bin $t/x/node && sudo -n tar -xzf $t/c.tgz -C $t/x --strip-components=1 && "
            f"sudo -n tar -xzf $t/n.tgz -C $t/x/node --strip-components=1 && "
            f"printf {shlex.quote(wrapper)} | sudo -n tee $t/x/bin/codex >/dev/null && "
            f"sudo -n chmod -R a+rX $t/x && sudo -n chmod 755 $t/x/bin/codex && "
            f"sudo -n mkdir -p {CODEX_CACHE_ROOT} && sudo -n mv -T $t/x {d}; "
            f"rc=$?; sudo -n rm -rf $t; [ $rc = 0 ]; }}; }}"
        )


# On a cache hit the container must end up with the same commands, in the same
# places, as Harbor's own codex install (agents/installed/codex.py).  Measured
# on boat (SESSION26/logs/E8/tools-compare-merge2.jsonl): on glibc images
# Harbor leaves curl and ripgrep on the plain PATH (what a verifier's test.sh
# sees), while node, npm and codex live only under ``~/.nvm`` and are on PATH
# only after ``. ~/.nvm/nvm.sh`` (what Harbor's codex run does).  Harbor's
# root symlink step finds nothing because root's shell never loads nvm.
#
# Step 1, CODEX_CACHE_SETUP, as root:
# * rg: from the Codex bundle, when the image lacks it;
# * curl: installed the way Harbor installs it, when the image lacks it;
# * musl (Alpine): ``apk add nodejs npm`` as Harbor does, and codex on the
#   plain PATH (Harbor's ``npm install -g`` puts it there);
# * glibc: only check that the cached Node runs (the image's glibc may be too old).
# It prints ``yeto-libc=musl`` or ``yeto-libc=glibc`` last.
# Step 2, CODEX_CACHE_NVM (glibc only), as the task's default user: a minimal
# ``~/.nvm`` whose nvm.sh puts the cached node, npm, npx and codex on PATH.
# Either step failing leaves no codex reachable, so Harbor's version probe
# misses and it runs its full install.
_MUSL = "/opt/yeto-codex/vendor/x86_64-unknown-linux-musl"
_NODE = "/opt/yeto-codex/node/bin"
_LIBC_MARK = "yeto-libc="
CODEX_CACHE_SETUP = (
    "set -e; mkdir -p /usr/local/bin; "
    "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then musl=1; else musl=0; fi; "
    "command -v rg >/dev/null 2>&1 || ln -sf " + _MUSL + "/codex-path/rg /usr/local/bin/rg; "
    "if ! command -v curl >/dev/null 2>&1; then "
    "if [ $musl = 1 ]; then apk add --no-cache curl; "
    "elif command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y curl; "
    "elif command -v yum >/dev/null 2>&1; then yum install -y curl; fi; fi; "
    "command -v curl >/dev/null 2>&1 && command -v rg >/dev/null 2>&1 || exit 1; "
    "if [ $musl = 1 ]; then "
    "{ command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; } || apk add --no-cache nodejs npm; "
    "node --version >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && "
    "ln -sf /opt/yeto-codex/bin/codex /usr/local/bin/codex || { rm -f /usr/local/bin/codex; exit 1; }; "
    "echo " + _LIBC_MARK + "musl; "
    "else " + _NODE + "/node --version >/dev/null 2>&1 || exit 1; echo " + _LIBC_MARK + "glibc; fi"
)


def codex_cache_nvm(node_version: str) -> str:
    """Step 2 (glibc): a minimal ``~/.nvm`` for the default user; refuses an image that already has one."""
    bin_rel = f"versions/node/v{node_version}/bin"
    nvm_sh = (f'export NVM_DIR="$HOME/.nvm"\n'
              f'case ":$PATH:" in *":$NVM_DIR/{bin_rel}:"*) ;; *) PATH="$NVM_DIR/{bin_rel}:$PATH" ;; esac\n'
              f"export PATH\n")
    return (
        'set -e; [ -n "$HOME" ] && [ ! -e "$HOME/.nvm" ] || exit 1; t="$HOME/.nvm.yeto-tmp"; rm -rf "$t"; '
        f'mkdir -p "$t/{bin_rel}"; '
        f'for b in node npm npx; do [ -e {_NODE}/$b ] && ln -s {_NODE}/$b "$t/{bin_rel}/$b"; done; '
        f'ln -s /opt/yeto-codex/bin/codex "$t/{bin_rel}/codex"; '
        f"printf {shlex.quote(nvm_sh)} > \"$t/nvm.sh\"; "
        'mv -T "$t" "$HOME/.nvm"; '
        '( . "$HOME/.nvm/nvm.sh" && node --version && command -v npm && codex --version ) >/dev/null 2>&1 '
        '|| { rm -rf "$HOME/.nvm"; exit 1; }'
    )


CODEX_CACHE_ROOT = "/var/cache/yeto-codex"  # outside the staging dir: recycle keeps it
CODEX_MOUNT = "/opt/yeto-codex"


def pinned_codex_cache() -> CodexCache:
    from .pins import (
        CODEX_LINUX_X64_SHA512,
        CODEX_LINUX_X64_URL,
        CODEX_VERSION,
        NODE_LINUX_X64_SHA256,
        NODE_LINUX_X64_URL,
        NODE_VERSION,
    )

    return CodexCache(version=CODEX_VERSION, url=CODEX_LINUX_X64_URL, sha512=CODEX_LINUX_X64_SHA512,
                      node_version=NODE_VERSION, node_url=NODE_LINUX_X64_URL, node_sha256=NODE_LINUX_X64_SHA256)


@dataclass(frozen=True)
class ExecOutput:
    stdout: str
    stderr: str
    return_code: int


class BoatSandbox:
    def __init__(self, client: BoatClient, quota: BoatQuota, *, name: str,
                 machine_type: str = "small", ttl_s: int = DEFAULT_TTL_S,
                 ready_timeout_s: float = 300.0, pull_timeout_s: float = 1800.0,
                 poll_interval_s: float = 1.0, ready_poll_s: float = 0.5, codex_cache: CodexCache | None = None,
                 sleep: Callable[[float], object] = asyncio.sleep) -> None:
        self._codex_cache = codex_cache
        self._client = client
        self._quota = quota
        self._name = name
        self._machine_type = machine_type
        self._ttl_s = ttl_s
        self._ready_timeout_s = ready_timeout_s
        self._pull_timeout_s = pull_timeout_s
        self._poll_s = poll_interval_s
        self._ready_poll_s = ready_poll_s
        self._sleep = sleep
        self._holds_slot = False
        self.sandbox_id: str | None = None
        self.usage_seconds: float | None = None
        self.machine_type = machine_type
        self.image: str | None = None  # image of the current (or last) task container
        self.timings: dict[str, float] = {}
        self.reuse_count = 0

    @property
    def name(self) -> str:
        return self._name

    # -- lifecycle -----------------------------------------------------------

    async def start(self, spec: ContainerSpec) -> None:
        await self._start(spec, run=True)

    async def start_vm(self, spec: ContainerSpec | None = None) -> None:
        """Start the VM without a task container; pull ``spec.image`` when given (pool prewarm)."""
        await self._start(spec, run=False)

    async def _start(self, spec: ContainerSpec | None, *, run: bool) -> None:
        if self._holds_slot:
            raise BoatError("boat sandbox already started")
        self.timings = {}
        await self._quota.acquire()
        self._holds_slot = True
        try:
            started = time.monotonic()
            await self._create()
            self.timings["create_s"] = time.monotonic() - started
            logger.info("boat sandbox %s created for %s (%s)", self.sandbox_id, self._name,
                        self._machine_type)
            started = time.monotonic()
            await self._wait_ready()
            self.timings["ready_s"] = time.monotonic() - started
            if run:
                assert spec is not None
                await self._run_container(spec)
            elif spec is not None:
                await self._pull(spec.image)
        except BaseException:
            await self.close()
            raise

    async def _create(self) -> None:
        # The create runs in a thread that a cancellation cannot stop.  On cancel,
        # wait for that thread so its sandbox id is known and close() deletes the
        # VM; otherwise it would run unseen until its TTL.
        create = asyncio.ensure_future(asyncio.to_thread(
            self._client.create, machine_type=self._machine_type, ttl_seconds=self._ttl_s,
            idempotency_key=f"yeto-{self._name}-{uuid.uuid4().hex}"))
        try:
            self.sandbox_id = await asyncio.shield(create)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                self.sandbox_id = await create
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
            await self._sleep(self._ready_poll_s)

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
        cache = self._codex_cache
        if cache is None:
            fetch, wait = "", "m= && c=0 && "
        else:
            # The fetch runs beside the pull; if it fails the task runs without the
            # cache and Harbor installs Codex itself (slower, same version).
            fetch = f"( {cache.fetch_command()} ) >/tmp/yeto-codex-fetch.log 2>&1 & f=$! ; "
            wait = (f"if wait $f; then m={shlex.quote(f'-v {cache.host_dir}:{CODEX_MOUNT}:ro')}; c=1; "
                    f"else m=; c=0; fi && ")
            run += ["$m"]
        run += [image, "sh", "-c", shlex.quote("sleep infinity")]
        # One VM command (one poll loop); the VM clock splits the phases.
        command = (f"{fetch}s=$(date +%s%N) && {DOCKER} pull -q {image} >/dev/null && p=$(date +%s%N) && "
                   f"{wait}q=$(date +%s%N) && {' '.join(run)} >/dev/null && e=$(date +%s%N) && "
                   f"echo {_PHASE_MARK} $((p-s)) $((e-q)) $((q-p)) $c")
        started = time.monotonic()
        result = await self.vm_exec(command, timeout_s=self._pull_timeout_s)
        if result.return_code != 0:
            raise BoatError(f"docker run {spec.image} failed ({result.return_code}): "
                            f"{(result.stderr or result.stdout)[-2000:]}")
        self.timings["container_s"] = time.monotonic() - started
        self.timings.update(_parse_phase_mark(result.stdout))
        self.image = spec.image
        if self.timings.get("codex_cache") == 1.0:
            started = time.monotonic()
            linked = await self.exec(CODEX_CACHE_SETUP, user="root", timeout_s=600,
                                     env={"DEBIAN_FRONTEND": "noninteractive"})
            if linked.return_code == 0 and f"{_LIBC_MARK}glibc" in linked.stdout:
                linked = await self.exec(codex_cache_nvm(cache.node_version), timeout_s=60)
            self.timings["codex_link_s"] = time.monotonic() - started
            if linked.return_code != 0:
                # No codex on PATH now: Harbor runs its full install (curl and ripgrep included).
                logger.warning("boat Codex cache: setup failed on %s; Harbor will install Codex: %s",
                               self.sandbox_id, (linked.stderr or linked.stdout)[-500:])
                self.timings["codex_cache"] = 0.0
        elif cache is not None:
            logger.warning("boat Codex cache: fetch failed on %s (see /tmp/yeto-codex-fetch.log on the VM); "
                           "Harbor will install Codex", self.sandbox_id)
        if spec.workdir:
            await self.exec(f"mkdir -p {shlex.quote(spec.workdir)}", user="root", timeout_s=60)

    async def _pull(self, image: str) -> None:
        started = time.monotonic()
        await self._checked(f"{DOCKER} pull -q {shlex.quote(image)} >/dev/null", f"docker pull {image}",
                            timeout_s=self._pull_timeout_s)
        self.timings["pull_s"] = time.monotonic() - started
        self.image = image

    # -- reuse (S26 E8) -----------------------------------------------------

    async def recycle(self) -> None:
        """Remove the task container and every trace of the task on the VM.

        Removed: all containers (with their anonymous volumes), all volumes,
        user-defined networks, the staging directory.  Kept: the pulled
        images (the reason to reuse) and the VM itself.  Raises
        :class:`BoatError` when the VM still shows task state afterwards;
        the caller must then :meth:`close` it instead of reusing it.
        """
        self._require_started()
        await self._checked(
            f"ids=$({DOCKER} ps -aq) && {{ [ -z \"$ids\" ] || {DOCKER} rm -f -v $ids >/dev/null; }} && "
            f"{DOCKER} volume prune -af >/dev/null && {DOCKER} network prune -f >/dev/null && "
            f"sudo -n rm -rf {STAGING_DIR}", "recycle", timeout_s=120.0)
        await self._verify_clean()

    async def _verify_clean(self) -> None:
        result = await self.vm_exec(
            f"echo c=$({DOCKER} ps -aq | wc -l) v=$({DOCKER} volume ls -q | wc -l) "
            f"s=$([ -e {STAGING_DIR} ] && echo 1 || echo 0)", timeout_s=60.0)
        if result.return_code != 0 or result.stdout.split() != ["c=0", "v=0", "s=0"]:
            raise BoatError(f"boat sandbox {self.sandbox_id} not clean after recycle: "
                            f"{(result.stdout or result.stderr)[-500:]!r}")

    async def reuse(self, spec: ContainerSpec, *, name: str, codex_cache: CodexCache | None = None) -> None:
        """Run a fresh task container on this (recycled) VM for the task ``name``.

        The VM must have passed :meth:`recycle`.  The new container starts
        from the image, so no file of the previous task survives.  On
        failure the VM is deleted, as in :meth:`start`.
        """
        self._require_started()
        self._name = name
        self._codex_cache = codex_cache  # the new task's choice (a cached build stays on the VM either way)
        self.timings = {"create_s": 0.0, "ready_s": 0.0}
        try:
            await self._verify_clean()
            await self._run_container(spec)
        except BaseException:
            await self.close()
            raise
        self.reuse_count += 1

    async def close(self) -> None:
        """Delete the VM (idempotent).  Errors are logged: the caller is often unwinding."""
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        try:
            if sandbox_id is not None:
                # One shielded step: a cancel during the usage read must not skip the delete.
                await asyncio.shield(self._read_usage_and_delete(sandbox_id))
        except Exception:
            # ttlSeconds still stops it; `boat list` shows it for manual cleanup.
            logger.exception("boat sandbox %s delete failed; delete it by hand", sandbox_id)
        finally:
            if self._holds_slot:
                self._holds_slot = False
                self._quota.release()

    async def _read_usage_and_delete(self, sandbox_id: str) -> None:
        try:
            self.usage_seconds = await asyncio.to_thread(self._client.usage_seconds, sandbox_id)
        except Exception:  # usage is a metric; never block the delete
            logger.warning("boat usage read failed for %s", sandbox_id)
        await asyncio.to_thread(self._client.delete, sandbox_id)
        logger.info("boat sandbox %s deleted (%s billable s)", sandbox_id, self.usage_seconds)

    # -- exec ----------------------------------------------------------------

    async def vm_exec(self, command: str, *, timeout_s: float | None) -> ExecOutput:
        """Run ``command`` on the VM; past ``timeout_s`` it is stopped and RuntimeError raised."""
        sandbox_id = self._require_started()
        start = asyncio.ensure_future(asyncio.to_thread(self._client.start_command, sandbox_id, command))
        try:
            process_id = await asyncio.shield(start)
        except asyncio.CancelledError:
            # The command may already be running on the VM: stop it once its id is known.
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
        return await self._then_clean(asyncio.to_thread(self._client.get_file, sandbox_id, staged),
                                      f"sudo -n rm -f {s}")

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        sandbox_id = self._require_started()
        staged = self._staging()
        s = shlex.quote(staged)
        await self._checked(
            f"mkdir -p {s} && {DOCKER} cp {CONTAINER}:{shlex.quote(source_dir.rstrip('/') + '/.')} {s} && "
            f"sudo -n tar -czf {s}.tar.gz -C {s} . && sudo -n chmod a+r {s}.tar.gz",
            f"download dir {source_dir}")
        archive = await self._then_clean(asyncio.to_thread(self._client.get_file, sandbox_id, f"{staged}.tar.gz"),
                                         f"sudo -n rm -rf {s} {s}.tar.gz")
        await asyncio.to_thread(extract_dir, archive, target_dir)

    async def _then_clean(self, work: Awaitable[T], cleanup: str) -> T:
        """Await ``work``, then run the VM ``cleanup`` command.

        When ``work`` raises, a failing cleanup is logged and the original
        error propagates (a ``finally`` would replace it with the cleanup's).
        """
        try:
            result = await work
        except BaseException:
            try:
                await self.vm_exec(cleanup, timeout_s=60)
            except Exception:  # the work's error is the one to report
                logger.warning("boat staging cleanup failed after an error: %s", cleanup, exc_info=True)
            raise
        await self.vm_exec(cleanup, timeout_s=60)
        return result

    def _require_started(self) -> str:
        if self.sandbox_id is None:
            raise BoatError("boat sandbox is not started (or already deleted)")
        return self.sandbox_id


_PHASE_MARK = "yeto-boat-phases"


def _parse_phase_mark(stdout: str) -> dict[str, float]:
    """``pull_s``/``run_s`` from the VM's phase line; empty when it is missing or garbled."""
    for line in reversed(stdout.splitlines()):
        fields = line.split()
        if len(fields) in (3, 5) and fields[0] == _PHASE_MARK:
            try:
                out = {"pull_s": int(fields[1]) / 1e9, "run_s": int(fields[2]) / 1e9}
                if len(fields) == 5:  # codex_fetch_wait_s: fetch time not hidden behind the pull
                    out.update(codex_fetch_wait_s=int(fields[3]) / 1e9, codex_cache=float(int(fields[4])))
                return out
            except ValueError:
                return {}
    return {}


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

