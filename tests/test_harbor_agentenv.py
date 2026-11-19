"""CPU tests of the AgentENV sandbox provider with a fake E2B-compatible client."""

from __future__ import annotations

import asyncio
import io
import re
import tarfile
import threading
from pathlib import Path

import pytest

from tests.sleep_fakes import fake_module_sleep
from yeto.rl.harness.harbor import agentenv_client
from yeto.rl.harness.harbor.agentenv_client import (
    AgentenvError,
    AgentenvQuota,
    CommandStatus,
    check_in_flight_cap,
)
from yeto.rl.harness.harbor.agentenv_sandbox import AgentenvSandbox, SandboxSpec, wrap_bash

# agentenv_env imports harbor at module load; reference its tuning knob by name so
# collection does not require harbor (the env tests pytest.importorskip it).
REACH_TRIES_ENV = "YETO_AGENTENV_REACH_TRIES"


def _tar_of(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeAgentenv:
    """In-memory AgentENV account: microVMs, background commands, guest files."""

    def __init__(self, *, fail_on: str | None = None, hang_on: str | None = None,
                 create_raises: bool = False, curl_code: int | None = 200,
                 curl_exit: int = 0) -> None:
        self.fail_on = fail_on
        self.hang_on = hang_on
        self.create_raises = create_raises
        # The in-microVM reachability self-check probes each endpoint with curl;
        # emulate a reply (``curl_code``, exit 0) or a network failure (``curl_exit``).
        self.curl_code = curl_code
        self.curl_exit = curl_exit
        self.live: set[str] = set()
        self.deleted: list[str] = []
        self.ttls: list[tuple[str, int]] = []
        self.commands: list[str] = []
        self.command_user: list[str | int | None] = []
        self.stopped: list[int] = []
        self.files: dict[str, bytes] = {}
        self.templates: list[str] = []
        self.forks: list[str] = []
        self.created = 0

    def create(self, *, template: str, ttl_seconds: int, metadata, env, idempotency_key: str) -> str:
        if self.create_raises:
            raise AgentenvError("agentenv create failed: SandboxException")
        self.created += 1
        sandbox_id = f"sb_{self.created}"
        self.live.add(sandbox_id)
        self.templates.append(template)
        return sandbox_id

    def fork(self, sandbox_id: str, count: int, *, ttl_seconds: int) -> list[str]:
        assert sandbox_id in self.live  # the seed stays live and unchanged
        self.forks.append(sandbox_id)
        ids = []
        for _ in range(count):
            self.created += 1
            fid = f"fk_{self.created}"
            self.live.add(fid)
            ids.append(fid)
        return ids

    def start_command(self, sandbox_id: str, command: str, *, cwd, env, user) -> int:
        assert sandbox_id in self.live
        self.commands.append(command)
        self.command_user.append(user)
        if (m := re.search(r"tar -czf (\S+\.tar\.gz)", command)):
            self.files[m.group(1)] = _tar_of({"./reward.txt": b"1\n"})
        return len(self.commands)

    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus:
        command = self.commands[process_id - 1]
        if self.hang_on and self.hang_on in command:
            return CommandStatus(running=True, exit_code=None, stdout="", stderr="")
        if self.fail_on and self.fail_on in command:
            return CommandStatus(running=False, exit_code=125, stdout="", stderr="boom")
        if "curl" in command:  # reachability self-check probe
            if self.curl_exit != 0:
                return CommandStatus(running=False, exit_code=self.curl_exit, stdout="",
                                     stderr=f"curl: ({self.curl_exit}) could not connect")
            return CommandStatus(running=False, exit_code=0,
                                 stdout=f"{self.curl_code} 0.012", stderr="")
        if "echo w=" in command:
            return CommandStatus(running=False, exit_code=0, stdout="w=0 s=0\n", stderr="")
        return CommandStatus(running=False, exit_code=0, stdout="ok\n", stderr="")

    def stop_command(self, sandbox_id: str, process_id: int) -> None:
        self.stopped.append(process_id)

    def put_file(self, sandbox_id: str, path: str, data: bytes, *, user) -> None:
        assert sandbox_id in self.live
        self.files[path] = data

    def get_file(self, sandbox_id: str, path: str, *, user) -> bytes:
        return self.files[path]

    def usage_seconds(self, sandbox_id: str) -> float | None:
        return None

    def set_ttl(self, sandbox_id: str, ttl_seconds: int) -> None:
        assert sandbox_id in self.live
        self.ttls.append((sandbox_id, ttl_seconds))

    def delete(self, sandbox_id: str) -> None:
        self.live.discard(sandbox_id)
        self.deleted.append(sandbox_id)


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def _sandbox(fake: FakeAgentenv, quota: AgentenvQuota | None = None, **kwargs) -> AgentenvSandbox:
    return AgentenvSandbox(fake, quota or AgentenvQuota(2, 30), name="task__abc__env", sleep=_no_sleep,
                          poll_interval_s=0.0, **kwargs)


SPEC = SandboxSpec(template="ubuntu:22.04", workdir="/app", cpus=1, memory_mb=2048)


def test_lifecycle_runs_directly_in_the_microvm_and_deletes_it():
    fake = FakeAgentenv()
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        out = await sandbox.exec("echo hi", cwd="/app", env={"A": "x y"}, user="root", timeout_s=30)
        await sandbox.close()
        await sandbox.close()  # idempotent
        return out

    out = asyncio.run(go())
    assert out.return_code == 0 and out.stdout == "ok\n"
    assert fake.templates == ["ubuntu:22.04"]  # microVM boots from the template, no docker run
    assert not any("docker" in c for c in fake.commands)  # no docker layer
    assert fake.commands[0].startswith("mkdir -p /app")  # workdir prepared directly in the guest
    assert fake.commands[-1] == wrap_bash("echo hi") and fake.command_user[-1] == "root"
    assert fake.deleted == ["sb_1"] and not fake.live and sandbox.sandbox_id is None


def test_create_failure_deletes_nothing_and_frees_the_slot():
    fake = FakeAgentenv(create_raises=True)
    quota = AgentenvQuota(1, 30)
    with pytest.raises(AgentenvError, match="create failed"):
        asyncio.run(_sandbox(fake, quota).start(SPEC))
    assert fake.created == 0 and not fake.live
    # The single slot is free again: a second start does not wait.
    asyncio.run(asyncio.wait_for(_sandbox(FakeAgentenv(), quota).start(SPEC), timeout=5))


def test_failed_workdir_prepare_deletes_the_microvm():
    fake = FakeAgentenv(fail_on="mkdir -p /app")
    with pytest.raises(AgentenvError, match="prepare workdir"):
        asyncio.run(_sandbox(fake).start(SPEC))
    assert fake.deleted == ["sb_1"]


def test_exec_timeout_stops_the_command():
    fake = FakeAgentenv(hang_on="sleep 999")
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await sandbox.exec("sleep 999", timeout_s=0)
        finally:
            await sandbox.close()

    asyncio.run(go())
    assert fake.stopped == [len(fake.commands)] and fake.deleted == ["sb_1"]


def test_cancelled_exec_stops_the_command_and_close_still_deletes():
    fake = FakeAgentenv(hang_on="agent-run")
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        task = asyncio.create_task(sandbox.exec("agent-run"))
        while not any("agent-run" in command for command in fake.commands):
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sandbox.close()

    asyncio.run(go())
    assert fake.stopped and fake.deleted == ["sb_1"]


def test_cancel_while_the_command_starts_still_stops_it():
    fake = FakeAgentenv(hang_on="agent-run")
    sandbox = _sandbox(fake)
    started, release = threading.Event(), threading.Event()
    real_start = fake.start_command

    def slow_start(sandbox_id, command, *, cwd, env, user):
        if "agent-run" in command:
            started.set()
            release.wait(5)
        return real_start(sandbox_id, command, cwd=cwd, env=env, user=user)

    fake.start_command = slow_start

    async def go():
        await sandbox.start(SPEC)
        task = asyncio.create_task(sandbox.exec("agent-run"))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sandbox.close()

    asyncio.run(go())
    agent_pid = next(i for i, command in enumerate(fake.commands, 1) if "agent-run" in command)
    assert fake.stopped == [agent_pid] and fake.deleted == ["sb_1"]


def test_cancel_during_create_still_deletes_the_microvm():
    gate = threading.Event()

    class SlowCreate(FakeAgentenv):
        def create(self, **kwargs) -> str:
            gate.wait(5)
            return super().create(**kwargs)

    fake = SlowCreate()
    sandbox = _sandbox(fake)

    async def go():
        task = asyncio.create_task(sandbox.start(SPEC))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        gate.set()  # the create returns after the cancel
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert fake.created == 1 and fake.deleted == ["sb_1"] and not fake.live


def test_file_transfer_is_direct_with_dirs_as_one_tar(tmp_path: Path):
    fake = FakeAgentenv()
    sandbox = _sandbox(fake)
    src = tmp_path / "tests"
    src.mkdir()
    (src / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (src / "test.sh").chmod(0o755)

    async def go():
        await sandbox.start(SPEC)
        await sandbox.upload_bytes(b"payload", "/solution/solve.sh")
        await sandbox.upload_dir(src, "/tests")
        await sandbox.download_dir("/logs/verifier", tmp_path / "out")
        await sandbox.close()

    asyncio.run(go())
    assert fake.files["/solution/solve.sh"] == b"payload"  # written straight into the guest
    assert (tmp_path / "out" / "reward.txt").read_text() == "1\n"
    staged = [p for p in fake.files if p.endswith(".tar.gz") and "/tmp/yeto-xfer/" in p]
    assert staged, "directories travel as one tar archive under /tmp"
    uploaded = tarfile.open(fileobj=io.BytesIO(fake.files[staged[0]]))
    assert uploaded.getmember("./test.sh").mode & 0o111  # exec bit survives
    assert any("tar -xzf" in c and "/tests" in c for c in fake.commands)
    assert any("rm -f" in c for c in fake.commands)  # staging is cleaned up


def test_wrap_bash_quotes_the_command():
    assert wrap_bash("echo '$HOME'; rm -rf /") == "bash -c 'echo '\"'\"'$HOME'\"'\"'; rm -rf /'"


def test_quota_limits_starts_per_minute(monkeypatch):
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    fake_module_sleep(monkeypatch, agentenv_client, fake_sleep, name="asyncio")
    quota = AgentenvQuota(10, 2, clock=lambda: now[0])

    async def go():
        for _ in range(3):
            await quota.acquire()

    asyncio.run(go())
    assert slept == [60.0]


def test_quota_active_cap_blocks_until_release():
    quota = AgentenvQuota(1, 30)

    async def go():
        await quota.acquire()
        waiter = asyncio.create_task(quota.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        quota.release()
        await asyncio.wait_for(waiter, timeout=5)

    asyncio.run(go())
    with pytest.raises(ValueError):
        AgentenvQuota(0, 30)


def test_in_flight_cap_matches_the_cap(monkeypatch):
    cap = agentenv_client.MAX_ACTIVE_SANDBOXES
    assert check_in_flight_cap(cap) == cap
    with pytest.raises(ValueError, match=f"active-sandbox cap {cap}"):
        check_in_flight_cap(cap + 1)
    monkeypatch.setenv(agentenv_client.MAX_ACTIVE_ENV, "8")
    with pytest.raises(ValueError):
        check_in_flight_cap(9)


def test_token_comes_from_either_env_name(monkeypatch):
    monkeypatch.delenv(agentenv_client.API_KEY_ENV, raising=False)
    monkeypatch.delenv(agentenv_client.SDK_API_KEY_ENV, raising=False)
    with pytest.raises(AgentenvError) as err:
        agentenv_client.load_token()
    assert "secret" not in str(err.value).lower() and "e2b_" not in str(err.value)
    monkeypatch.setenv(agentenv_client.SDK_API_KEY_ENV, "e2b_fallback")
    assert agentenv_client.load_token() == "e2b_fallback"
    monkeypatch.setenv(agentenv_client.API_KEY_ENV, "agentenv_primary")
    assert agentenv_client.load_token() == "agentenv_primary"


# -- Harbor side (needs the pinned Harbor installed) --------------------------


def test_agentenv_task_maps_to_the_import_path(tmp_path: Path):
    pytest.importorskip("harbor")
    from yeto.rl.agentic import AgentTask
    from yeto.rl.harness.harbor import HarborSettings, build_trial_config
    from yeto.rl.harness.harbor.agentenv_env import AGENTENV_IMPORT_PATH

    (tmp_path / "t1").mkdir()
    task = AgentTask.from_mapping({"task_id": "t1", "agent": "oracle", "provider": "agentenv",
                                   "model_name": "m", "wall_clock_s": 60.0})
    config = build_trial_config(task, "http://x", HarborSettings(tasks_dir=tmp_path))
    assert config.environment.import_path == AGENTENV_IMPORT_PATH and config.environment.type is None


def test_agentenv_environment_on_harbor_interface(tmp_path: Path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor.agentenv_env import AgentenvEnvironment

    fake = FakeAgentenv()
    env = AgentenvEnvironment(
        environment_dir=tmp_path, environment_name="t1", session_id="t1__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="img:1", memory_mb=2048, cpus=1),
        client=fake, quota=AgentenvQuota(2, 30))
    assert env.type() == "agentenv" and not env.capabilities.mounted

    async def go():
        await env.start(force_build=False)
        result = await env.exec("true", timeout_sec=10)
        await env.stop(delete=True)
        return result

    result = asyncio.run(go())
    assert result.return_code == 0 and fake.templates == ["img:1"] and fake.deleted == ["sb_1"]
    with pytest.raises(ValueError, match="docker_image"):
        AgentenvEnvironment(environment_dir=tmp_path, environment_name="t1", session_id="s",
                            trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
                            task_env_config=EnvironmentConfig(), client=fake)


def _reach_env(tmp_path: Path, fake: FakeAgentenv, **reach_kwargs):
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor.agentenv_env import AgentenvEnvironment

    return AgentenvEnvironment(
        environment_dir=tmp_path, environment_name="t1", session_id="t1__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="img:1", memory_mb=2048, cpus=1),
        client=fake, quota=AgentenvQuota(2, 30), **reach_kwargs)


def test_agentenv_network_check_raises_when_judge_unreachable(tmp_path: Path, monkeypatch):
    """P0-1: an unreachable required endpoint fails the start loud, not silently."""
    pytest.importorskip("harbor")
    from yeto.rl.harness.harbor.agentenv_network import NetworkError

    monkeypatch.setenv(REACH_TRIES_ENV, "1")  # no retry backoff in the test
    # curl inside the microVM cannot connect to the judge/records endpoint.
    fake = FakeAgentenv(curl_exit=7)
    env = _reach_env(tmp_path, fake,
                     reach_endpoints={"task-server": "http://judge.invalid:8000",
                                      "records-proxy": "http://records.invalid:9000"})

    with pytest.raises(NetworkError, match="required sidecar"):
        asyncio.run(env.start(force_build=False))
    assert fake.deleted == ["sb_1"]  # the microVM is torn down on a failed self-check


def test_agentenv_network_check_passes_when_reachable(tmp_path: Path, monkeypatch):
    """A reachable endpoint lets the start proceed (and the probe actually ran)."""
    pytest.importorskip("harbor")

    monkeypatch.setenv(REACH_TRIES_ENV, "1")
    fake = FakeAgentenv(curl_code=200)
    env = _reach_env(tmp_path, fake, reach_endpoints={"records-proxy": "http://records.ok:9000"})

    async def go():
        await env.start(force_build=False)
        await env.stop(delete=True)

    asyncio.run(go())
    assert any("curl" in c for c in fake.commands)  # the reachability probe ran


def test_agentenv_network_check_discovers_records_proxy_from_process_env(tmp_path: Path, monkeypatch):
    """YETO_RECORDS_PROXY_PUBLIC_BASE in the sampler process env is probed (and must be reachable)."""
    pytest.importorskip("harbor")
    from yeto.rl.harness.harbor.agentenv_network import NetworkError

    monkeypatch.setenv(REACH_TRIES_ENV, "1")
    monkeypatch.setenv("YETO_RECORDS_PROXY_PUBLIC_BASE", "http://records.invalid:9000")
    fake = FakeAgentenv(curl_exit=7)  # unreachable
    env = _reach_env(tmp_path, fake)  # no explicit override: discover from env

    with pytest.raises(NetworkError, match="records-proxy"):
        asyncio.run(env.start(force_build=False))


def test_agentenv_network_check_noop_without_endpoints(tmp_path: Path, monkeypatch):
    """A self-contained task (no judge/records endpoint set) skips the probe cleanly."""
    pytest.importorskip("harbor")

    monkeypatch.delenv("YETO_RECORDS_PROXY_PUBLIC_BASE", raising=False)
    monkeypatch.delenv("TASK_SERVER_URL", raising=False)
    fake = FakeAgentenv(curl_exit=7)  # would fail IF a probe ran
    env = _reach_env(tmp_path, fake)

    async def go():
        await env.start(force_build=False)
        await env.stop(delete=True)

    asyncio.run(go())
    assert not any("curl" in c for c in fake.commands)  # nothing to probe, no probe ran
