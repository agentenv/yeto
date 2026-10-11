"""RLF Modal VM lifecycle + CTF environment orchestration (audit breakpoints 3/5).

Learner GPUs are deliberately never allocated here.

Two layers, both CPU-testable with a fake VM:

* VM lifecycle (S20): ``RlfModalVmConfig`` / ``RlfVmLease`` / ``acquire_vm``.
* CTF orchestration (breakpoint 3): the nested-Docker sequence verified in
  ``evidence/rlf/modal-ctf-trusted-recvfix2.log`` packaged as functions —
  dockerd readiness, ``docker build --network host``, target run with
  ``-e FLAG`` (env injection only), health wait, idempotent cleanup.

Transport decision (代拍板, rationale): the worker-facing wire is served by
``serve_environment`` **in the trusted process** and commands are relayed into
the VM through the Modal exec channel — the exact pattern ``tb2_provider``
uses — instead of running a token-protected exec HTTP service inside the VM.
Reasons: (1) the bearer token never enters the VM, so a compromised agent
command cannot read it; (2) nothing has to be baked into the VM image beyond
docker; (3) the worker speaks the unchanged ``HttpTerminalEnvironment`` wire.
The agent reaches the CTF target at ``127.0.0.1:31337`` because relay
commands exec in the VM's host network namespace and the target container
runs ``--network host``.

Flag hygiene: the FLAG reaches the target container only as ``-e FLAG`` on
``docker run``.  It is never written into stdout/stderr captures, logs, or
exception messages: every orchestration error passes through :func:`scrub`
with the flag, and :class:`OrchestrationError` never embeds the command line.

Cleanup order (each step idempotent): solver/grader containers -> target
container -> images are kept (layer cache for the next episode) -> VM
terminate.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import EpisodeLease

# ----------------------------------------------------------------------------- VM lifecycle (S20)


@dataclass(frozen=True)
class RlfModalVmConfig:
    app_name: str = 'yeto-rlf-env'
    cpu: float = 2.0
    memory_mb: int = 4096
    ttl_s: int = 1800
    image: str = 'debian:bookworm-slim'
    runtime: str = 'vm'
    image_run_commands: tuple = ()

    def validate(self):
        if self.runtime != 'vm':
            raise ValueError('RLF environment must use Modal VM runtime')
        if self.cpu <= 0 or self.memory_mb < 512:
            raise ValueError('invalid VM resources')


class RlfVmLease:
    def __init__(self, sandbox: Any, *, task_id: str, episode_id: str, trajectory_id: str | None = None):
        self.sandbox = sandbox
        self.task_id = task_id
        self.episode_id = episode_id
        self.closed = False
        self.trajectory_id = trajectory_id

    @property
    def object_id(self):
        return getattr(self.sandbox, 'object_id', '')

    def exec(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError('RLF VM lease is closed')
        return self.sandbox.exec(*args, **kwargs)

    def alive(self) -> bool:
        try:
            poll = getattr(self.sandbox, 'poll', None)
            return bool(poll) and poll() is None
        except Exception:  # noqa: BLE001 - a sandbox the API no longer knows is gone
            return False

    def teardown(self):
        if not self.closed:
            self.closed = True
            self.sandbox.terminate()


def acquire_vm(task_id: str, episode_id: str, *, config: RlfModalVmConfig | None = None, image: Any = None,
               trajectory_id: str | None = None) -> RlfVmLease:
    cfg = config or RlfModalVmConfig()
    cfg.validate()
    import modal
    app = modal.App.lookup(cfg.app_name, create_if_missing=True)
    if image is None:
        image = modal.Image.from_registry(cfg.image)
        if cfg.image_run_commands:
            image = image.run_commands(*cfg.image_run_commands)
    sb = modal.Sandbox.create('sleep', 'infinity', app=app, image=image, runtime='vm', cpu=cfg.cpu,
                              memory=cfg.memory_mb, timeout=cfg.ttl_s, idle_timeout=min(cfg.ttl_s, 300),
                              tags={'yeto-purpose': 'rlf-environment', 'yeto-task': task_id[:63],
                                    'yeto-episode': episode_id[:63]})
    return RlfVmLease(sb, task_id=task_id, episode_id=episode_id, trajectory_id=trajectory_id)


# ----------------------------------------------------------------------------- exec surface (fakeable)


@dataclass(frozen=True)
class VmExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


# One synchronous VM command: (shell command, timeout) -> captured result.
VmExec = Callable[..., VmExecResult]

_TIMEOUT_EXIT_CODES = frozenset({124, 137})


def _decode_output(data: bytes | str | None) -> str:
    if not data:
        return ''
    return data if isinstance(data, str) else data.decode('utf-8', errors='replace')


def vm_exec(lease: RlfVmLease) -> VmExec:
    """Adapt a Modal VM lease to the synchronous ``VmExec`` surface.

    Commands run as ``bash -lc`` in the VM's host network namespace (this is
    what lets the agent reach the ``--network host`` CTF target).
    """

    def run(command: str, *, timeout_s: float) -> VmExecResult:
        process = lease.exec('bash', '-lc', command, timeout=int(timeout_s) + 30, text=False)
        output = _decode_output(process.stdout.read()) + _decode_output(process.stderr.read())
        code = int(process.wait())
        return VmExecResult(code, output, code in _TIMEOUT_EXIT_CODES)

    return run


class OrchestrationError(RuntimeError):
    """A CTF orchestration step failed.  Messages are always flag-scrubbed."""


def scrub(text: Any, *secrets: str | None) -> str:
    """Replace every secret occurrence with ``***`` (defense in depth)."""
    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(str(secret), '***')
    return out


# ----------------------------------------------------------------------------- CTF task spec


@dataclass(frozen=True)
class RlfCtfTask:
    """One admitted CTF task package as the orchestrator needs it (trusted side)."""

    task_id: str
    package_dir: Path
    flag: str
    target_port: int = 31337
    grader_mode: str = 'flag'


def _task_yaml_field(text: str, name: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(name + ':'):
            return line.split(':', 1)[1].strip()
    return None


def resolve_ctf_task(task_id: str, tasks_dir: str | Path, flags: Mapping[str, str]) -> RlfCtfTask:
    """Fail closed: unknown task, missing package files, non-flag mode or a
    missing trusted flag all refuse the acquire (infrastructure, never a 0 reward)."""
    package_dir = Path(tasks_dir) / task_id
    yaml_path = package_dir / 'task.yaml'
    if not yaml_path.is_file():
        raise OrchestrationError(f'{task_id}: no task.yaml under {tasks_dir}')
    if not (package_dir / 'Dockerfile').is_file():
        raise OrchestrationError(f'{task_id}: package has no Dockerfile')
    text = yaml_path.read_text()
    mode = _task_yaml_field(text, 'grader_mode') or 'flag'
    if mode != 'flag':
        raise OrchestrationError(f'{task_id}: grader_mode {mode!r} is not packaged (flag only)')
    flag = flags.get(task_id)
    if not flag:
        raise OrchestrationError(f'{task_id}: no trusted flag configured')
    port = int(_task_yaml_field(text, 'target_port') or 31337)
    return RlfCtfTask(task_id=task_id, package_dir=package_dir, flag=str(flag), target_port=port)


# ----------------------------------------------------------------------------- dockerd / build / run / health

DOCKER_SOCK = 'unix:///run/docker/docker.sock'
_DOCKER = f'docker -H {DOCKER_SOCK}'
REMOTE_TASK_DIR = '/opt/task'
_STAGE_PATH = '/tmp/rlf-task.tgz.b64'
_STAGE_CHUNK = 48_000
# Only these package entries form the docker build context; ``private/``
# (reference solution) and admission metadata never enter the VM.
_BUILD_CONTEXT_ENTRIES = ('Dockerfile', 'entrypoint.sh', 'public')


def start_dockerd(exec_fn: VmExec, *, attempts: int = 40, sleep_s: float = 1.0,
                  sleep: Callable[[float], None] = time.sleep) -> None:
    """Reset and start nested dockerd, then wait for ``docker info``.

    Parameters verified on Modal (``modal-dockerd-probe.log`` /
    ``modal-ctf-trusted-recvfix2.log``): custom socket under /run/docker,
    ``--iptables=false --bridge=none`` (the VM has no bridge network).
    """
    result = exec_fn(
        'rm -rf /var/lib/docker /run/docker && mkdir -p /run/docker && '
        f'(dockerd --host={DOCKER_SOCK} --iptables=false --bridge=none >/tmp/dockerd.log 2>&1 &)',
        timeout_s=30,
    )
    if result.exit_code != 0:
        raise OrchestrationError(f'dockerd launch failed (rc={result.exit_code}): {result.output[-200:]}')
    for _ in range(max(1, attempts)):
        if exec_fn(f'{_DOCKER} info >/dev/null 2>&1', timeout_s=15).exit_code == 0:
            return
        sleep(sleep_s)
    raise OrchestrationError(f'dockerd did not become ready after {attempts} attempts')


def package_tar_b64(package_dir: str | Path) -> str:
    """tar+base64 of the build context only (never ``private/``)."""
    root = Path(package_dir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for entry in _BUILD_CONTEXT_ENTRIES:
            path = root / entry
            if path.is_file():
                tar.add(path, arcname=entry)
            elif path.is_dir():
                for item in sorted(path.rglob('*')):
                    if item.is_file() and '__pycache__' not in item.parts:
                        tar.add(item, arcname=str(item.relative_to(root)))
    return base64.b64encode(buf.getvalue()).decode()


def stage_package(exec_fn: VmExec, package_dir: str | Path, *, remote_dir: str = REMOTE_TASK_DIR) -> None:
    """Ship the build context into the VM (chunked base64, like the TB2 tests staging)."""
    b64 = package_tar_b64(package_dir)
    commands = [f'rm -rf {remote_dir} && mkdir -p {remote_dir} && : > {_STAGE_PATH}']
    commands += [f'printf %s {b64[i:i + _STAGE_CHUNK]} >> {_STAGE_PATH}'
                 for i in range(0, len(b64), _STAGE_CHUNK)]
    commands.append(f'base64 -d {_STAGE_PATH} | tar xzf - -C {remote_dir} && rm -f {_STAGE_PATH}')
    for command in commands:
        result = exec_fn(command, timeout_s=120)
        if result.exit_code != 0:
            raise OrchestrationError(f'staging the task package failed (rc={result.exit_code}): {result.output[-200:]}')


def build_target(exec_fn: VmExec, task: RlfCtfTask, *, image: str, remote_dir: str = REMOTE_TASK_DIR) -> None:
    """``docker build --network host`` — the default bridge hangs and
    ``--network=none`` fails in nested Docker (verified in recvfix2)."""
    result = exec_fn(f'{_DOCKER} build --network host -t {shlex.quote(image)} {shlex.quote(remote_dir)}',
                     timeout_s=600)
    if result.exit_code != 0:
        raise OrchestrationError(scrub(f'target image build failed (rc={result.exit_code}): '
                                       f'{result.output[-200:]}', task.flag))


def run_target(exec_fn: VmExec, task: RlfCtfTask, *, image: str, container: str) -> str:
    """Start the target with the FLAG as a container env var (the only place
    the flag crosses onto the VM).  Output/errors are scrubbed; the flag is
    never logged or raised."""
    command = (f'{_DOCKER} run -d --name {shlex.quote(container)} --network host '
               f'-e FLAG={shlex.quote(task.flag)} {shlex.quote(image)}')
    result = exec_fn(command, timeout_s=60)
    if result.exit_code != 0:
        raise OrchestrationError(scrub(f'target container failed to start (rc={result.exit_code}): '
                                       f'{result.output[-200:]}', task.flag))
    cid = result.output.strip().splitlines()[-1] if result.output.strip() else ''
    if not cid:
        raise OrchestrationError('target container start returned no container id')
    return cid


def wait_target_healthy(exec_fn: VmExec, task: RlfCtfTask, *, container: str, attempts: int = 30,
                        sleep_s: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
    """Block until the target's HEALTHCHECK reports healthy; fail fast on exit."""
    probe = (f"{_DOCKER} inspect -f '{{{{.State.Status}}}} "
             f"{{{{if .State.Health}}}}{{{{.State.Health.Status}}}}{{{{end}}}}' {shlex.quote(container)}")
    for _ in range(max(1, attempts)):
        result = exec_fn(probe, timeout_s=15)
        fields = result.output.split()
        status = fields[0] if fields else ''
        health = fields[1] if len(fields) > 1 else ''
        if status in ('exited', 'dead'):
            raise OrchestrationError(f'target container {container} is {status} before becoming healthy')
        if result.exit_code == 0 and (health == 'healthy' or (status == 'running' and not health)):
            return
        sleep(sleep_s)
    raise OrchestrationError(f'target container did not become healthy after {attempts} attempts')


def remove_containers(exec_fn: VmExec, containers: list[str] | tuple[str, ...]) -> None:
    """``docker rm -f`` in the given order; every step tolerates failure
    (idempotent teardown — a missing container is already clean)."""
    for name in containers:
        try:
            exec_fn(f'{_DOCKER} rm -f {shlex.quote(name)} >/dev/null 2>&1', timeout_s=30)
        except Exception:  # noqa: BLE001 - cleanup never masks the original outcome
            pass


# ----------------------------------------------------------------------------- orchestration state machine


@dataclass
class CtfOrchestration:
    """Verified nested-Docker sequence as an idempotent up/down state machine."""

    exec_fn: VmExec
    task: RlfCtfTask
    target_container: str = 'rlf-ctf-target'
    target_image: str = 'rlf-ctf-target'
    # Solver/grader containers (cleaned before the target); empty in the online
    # path (the agent relays into the VM host namespace and grading is
    # trusted-side), but the ordering contract is kept for reference-check runs.
    auxiliary_containers: tuple[str, ...] = ()
    dockerd_attempts: int = 40
    health_attempts: int = 30
    sleep: Callable[[float], None] = time.sleep
    _up_done: bool = False
    _down_done: bool = False

    def up(self) -> str:
        """dockerd -> stage -> build -> run(FLAG) -> health.  Returns the target cid."""
        if self._up_done:
            raise OrchestrationError('orchestration already brought up')
        try:
            start_dockerd(self.exec_fn, attempts=self.dockerd_attempts, sleep=self.sleep)
            stage_package(self.exec_fn, self.task.package_dir)
            build_target(self.exec_fn, self.task, image=self.target_image)
            cid = run_target(self.exec_fn, self.task, image=self.target_image, container=self.target_container)
            wait_target_healthy(self.exec_fn, self.task, container=self.target_container,
                                attempts=self.health_attempts, sleep=self.sleep)
        except BaseException:
            self.down()  # a half-up environment is always cleaned
            raise
        self._up_done = True
        return cid

    def down(self) -> None:
        """solver/grader containers -> target container; images stay; idempotent."""
        if self._down_done:
            return
        self._down_done = True
        remove_containers(self.exec_fn, [*self.auxiliary_containers, self.target_container])


# ----------------------------------------------------------------------------- relay environment + verifier


class CtfRelayEnvironment:
    """``TerminalEnvironment`` served in the trusted process; commands exec on the VM.

    One relay == one episode: the first episode id on the wire binds it
    (same contract as ``tb2_provider.EpisodeBinding``).  ``submit`` records
    the agent's submission for the trusted verifier; the flag comparison
    never happens here.
    """

    def __init__(self, exec_fn: VmExec) -> None:
        self._exec_fn = exec_fn
        self.episode_id: str | None = None
        self.commands: list[str] = []
        self.submissions: list[dict[str, Any]] = []
        self.submitted = False

    def _bind(self, episode_id: str) -> None:
        if self.episode_id is None:
            self.episode_id = episode_id
        elif episode_id != self.episode_id:
            from yeto.rl.harness.codex.client import EpisodeAPIError
            raise EpisodeAPIError(404, 'unknown_episode', 'unknown episode')

    async def execute(self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int):
        import asyncio
        self._bind(episode_id)
        if self.submitted:
            from yeto.rl.harness.codex.client import EpisodeAPIError
            raise EpisodeAPIError(409, 'episode_terminal', 'episode already submitted')
        self.commands.append(command)
        seconds = max(1, int(timeout_seconds))
        capped = (f'timeout -k 2 {seconds} bash -c {shlex.quote(command)} 2>&1 | '
                  f'head -c {int(output_bytes) + 1}; exit ${{PIPESTATUS[0]}}')
        result = await asyncio.to_thread(self._exec_fn, capped, timeout_s=timeout_seconds)
        truncated = len(result.output.encode('utf-8')) > output_bytes
        return {'exit_code': result.exit_code, 'output': result.output[:output_bytes],
                'timed_out': result.timed_out, 'truncated': truncated}

    async def submit(self, episode_id: str, submission: dict[str, Any]):
        self._bind(episode_id)
        self.submissions.append(dict(submission))
        self.submitted = True
        return {'accepted': True}


class RlfFlagVerifier:
    """Trusted-side flag grader (``grader_mode: flag``).

    Compares the agent's submitted flag against the trusted flag in constant
    time.  The evaluation payload carries the RLF grade under ``rlf_grade``
    (consumed by ``finish_trusted`` via ``trusted_grade_from_evaluation``)
    with the RLF identity quadruple in ``grader_evidence`` — the verifier is
    built per-lease, so that identity is trusted-side.  The flag itself never
    appears in the payload, the evidence, or the log.
    """

    def __init__(self, relay: CtfRelayEnvironment, *, flag: str, task_id: str, episode_id: str,
                 trajectory_id: str) -> None:
        self._relay = relay
        self._flag = flag
        self._identity = {'rlf_task_id': task_id, 'rlf_episode_id': episode_id,
                          'rlf_trajectory_id': trajectory_id}
        self.evaluations = 0

    async def evaluate(self, episode_id: str) -> dict[str, Any]:
        if self._relay.episode_id is not None and episode_id != self._relay.episode_id:
            raise ValueError(f'verifier bound to {self._relay.episode_id!r}, asked for {episode_id!r}')
        self.evaluations += 1
        import hmac as _hmac
        submission = self._relay.submissions[-1] if self._relay.submissions else None
        candidate = submission.get('flag') if isinstance(submission, Mapping) else None
        submitted = isinstance(candidate, str) and bool(candidate)
        matched = submitted and _hmac.compare_digest(candidate, self._flag)
        evidence = {'grader': 'rlf-flag', 'grader_mode': 'flag', 'submitted': submitted, **self._identity}
        grade: dict[str, Any] = {'status': 'completed', 'reward': 1.0 if matched else 0.0,
                                 'passed': bool(matched), 'grader_evidence': evidence}
        if not submitted:
            grade['error_class'] = 'missing_submission'
        return {'passed': bool(matched), 'rlf_grade': grade,
                'log': 'flag submission matched' if matched else 'flag submission missing or did not match'}


# ----------------------------------------------------------------------------- full environment lease


@dataclass(frozen=True)
class RlfCtfConfig:
    """CTF packaging for :class:`RlfModalProvider` (breakpoint 3)."""

    tasks_dir: str  # directory of admitted task packages (task_id -> package dir)
    flags: Mapping[str, str]  # task_id -> trusted flag (never leaves the trusted side)
    vm_image: str = 'docker:27-dind'  # nested Docker runtime (verified on Modal)
    lease_seconds: float = 1800.0
    dockerd_attempts: int = 40
    health_attempts: int = 30


@dataclass
class RlfCtfLease:
    """Full ``EnvironmentLease``: field names match the subprocess agent protocol."""

    env_url: str
    env_token: str
    verifier: RlfFlagVerifier
    destroy: Callable[[], Any]
    describe: Callable[[], Any]
    deadline_seconds: float
    worker_env: dict[str, str] | None = None
    task_id: str = ''
    episode_id: str = ''
    trajectory_id: str = ''
    vm_lease: RlfVmLease | None = None
    orchestration: CtfOrchestration | None = None
    environment: CtfRelayEnvironment | None = None


class RlfModalProvider:
    """Preflight-compatible RLF environment provider factory product.

    ``resolve_environment_provider`` calls ``factory(miles_args)`` and requires
    the result to expose ``acquire(task_id, trajectory_id)``; this class is that
    product.  Each acquire opens one ``EpisodeLease`` (RLF identity producer)
    and one isolated Modal VM bound to it.

    With ``ctf=None`` the returned lease is the S20 VM lease (exec/teardown +
    RLF identity; no HTTP endpoint/verifier).  With an :class:`RlfCtfConfig`
    each acquire additionally runs the packaged CTF orchestration
    (dockerd/target/flag/health) and returns a full :class:`RlfCtfLease`
    (env_url/env_token/verifier/destroy/describe/deadline) — audit
    breakpoint 3.
    """

    def __init__(self, *, config: RlfModalVmConfig | None = None, ctf: RlfCtfConfig | None = None):
        self.config = config or RlfModalVmConfig()
        self.ctf = ctf

    async def acquire(self, task_id: str, trajectory_id: str = '') -> RlfVmLease | RlfCtfLease:
        episode = EpisodeLease(task_id=task_id,
                               trajectory_id=trajectory_id or f'rlf-traj-{uuid.uuid4().hex}')
        if self.ctf is None:
            return acquire_vm(task_id, episode.episode_id, config=self.config,
                              trajectory_id=episode.trajectory_id)
        return await self._acquire_ctf(episode)

    async def _acquire_ctf(self, episode: EpisodeLease) -> RlfCtfLease:
        import asyncio
        import secrets

        from yeto.rl.harness.codex.environment import serve_environment

        ctf = self.ctf
        assert ctf is not None
        task = resolve_ctf_task(episode.task_id, ctf.tasks_dir, ctf.flags)
        # vm_exec/relay use ``bash -lc``; the default docker:27-dind image is
        # alpine-based and has no bash, so add it at image build time.
        vm_config = replace(self.config, image=ctf.vm_image,
                            image_run_commands=('apk add --no-cache bash',))
        vm = await asyncio.to_thread(acquire_vm, episode.task_id, episode.episode_id,
                                     config=vm_config, trajectory_id=episode.trajectory_id)
        orchestration = CtfOrchestration(vm_exec(vm), task,
                                         dockerd_attempts=ctf.dockerd_attempts,
                                         health_attempts=ctf.health_attempts)
        try:
            await asyncio.to_thread(orchestration.up)
            relay = CtfRelayEnvironment(vm_exec(vm))
            token = secrets.token_urlsafe(24)
            runner, url = await serve_environment(relay, token, host='127.0.0.1')
        except BaseException:
            orchestration.down()
            await asyncio.to_thread(vm.teardown)
            raise
        deadline = float(ctf.lease_seconds or self.config.ttl_s)
        state = {'destroyed': False}

        async def destroy() -> None:
            if state['destroyed']:
                return
            state['destroyed'] = True
            # solver/grader containers -> target container -> images stay -> VM terminate
            await asyncio.to_thread(orchestration.down)
            await runner.cleanup()
            await asyncio.to_thread(vm.teardown)

        async def describe() -> str:
            if not state['destroyed']:
                return 'live'
            for _ in range(20):
                if not await asyncio.to_thread(vm.alive):
                    return 'gone'
                await asyncio.sleep(0.25)
            return 'live'

        return RlfCtfLease(env_url=url, env_token=token,
                           verifier=RlfFlagVerifier(relay, flag=task.flag, task_id=episode.task_id,
                                                    episode_id=episode.episode_id,
                                                    trajectory_id=episode.trajectory_id),
                           destroy=destroy, describe=describe, deadline_seconds=deadline,
                           task_id=episode.task_id, episode_id=episode.episode_id,
                           trajectory_id=episode.trajectory_id, vm_lease=vm,
                           orchestration=orchestration, environment=relay)
