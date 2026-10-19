"""CPU tests of the boat sandbox provider (S25 N10.8) with a fake boat client."""

from __future__ import annotations

import asyncio
import io
import re
import tarfile
from pathlib import Path

import pytest

from yeto.rl.harness.harbor import boat_client
from yeto.rl.harness.harbor.boat_client import BoatError, BoatQuota, CommandStatus, check_in_flight_cap
from yeto.rl.harness.harbor.boat_sandbox import (
    BoatSandbox,
    ContainerSpec,
    docker_exec_command,
    extract_dir,
    machine_type_for,
    pack_dir,
)


def _tar_of(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeBoat:
    """In-memory boat account: VMs, detached commands, staged files."""

    def __init__(self, *, fail_on: str | None = None, hang_on: str | None = None,
                 states: list[str] | None = None) -> None:
        self.fail_on = fail_on
        self.hang_on = hang_on
        self.states = list(states or ["provisioning", "ready"])
        self.live: set[str] = set()
        self.deleted: list[str] = []
        self.commands: list[str] = []
        self.stopped: list[int] = []
        self.files: dict[str, bytes] = {}
        self.created = 0

    def create(self, *, machine_type: str, ttl_seconds: int, idempotency_key: str) -> str:
        self.created += 1
        sandbox_id = f"bx_{self.created}"
        self.live.add(sandbox_id)
        self.machine_type = machine_type
        return sandbox_id

    def state(self, sandbox_id: str) -> str:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def start_command(self, sandbox_id: str, command: str) -> int:
        assert sandbox_id in self.live
        self.commands.append(command)
        if (m := re.search(r"tar -czf (\S+\.tar\.gz)", command)):
            self.files[m.group(1)] = _tar_of({"./reward.txt": b"1\n"})
        if (m := re.search(r"docker cp -L main:\S+ (\S+)", command)):
            self.files[m.group(1)] = b"file-from-container"
        return len(self.commands)

    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus:
        command = self.commands[process_id - 1]
        if self.hang_on and self.hang_on in command:
            return CommandStatus(running=True, exit_code=None, stdout="", stderr="")
        if self.fail_on and self.fail_on in command:
            return CommandStatus(running=False, exit_code=125, stdout="", stderr="pull access denied")
        return CommandStatus(running=False, exit_code=0, stdout="ok\n", stderr="")

    def stop_command(self, sandbox_id: str, process_id: int) -> None:
        self.stopped.append(process_id)

    def put_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        self.files[path] = data

    def get_file(self, sandbox_id: str, path: str) -> bytes:
        return self.files[path]

    def usage_seconds(self, sandbox_id: str) -> float | None:
        return 42.0

    def delete(self, sandbox_id: str) -> None:
        self.live.discard(sandbox_id)
        self.deleted.append(sandbox_id)


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def _sandbox(fake: FakeBoat, quota: BoatQuota | None = None, **kwargs) -> BoatSandbox:
    return BoatSandbox(fake, quota or BoatQuota(2, 12), name="task__abc__env", sleep=_no_sleep,
                       poll_interval_s=0.0, **kwargs)


SPEC = ContainerSpec(image="alexgshaw/headless-terminal:20251031", workdir="/app", cpus=1, memory_mb=2048)


def test_lifecycle_runs_the_task_image_and_deletes_the_vm():
    fake = FakeBoat()
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        out = await sandbox.exec("echo hi", cwd="/app", env={"A": "x y"}, user="root", timeout_s=30)
        await sandbox.close()
        await sandbox.close()  # idempotent
        return out

    out = asyncio.run(go())
    assert out.return_code == 0 and out.stdout == "ok\n"
    run = fake.commands[0]
    assert "docker pull -q alexgshaw/headless-terminal:20251031" in run
    assert "--network host" not in run  # bridge: task servers must not collide with the VM's
    assert "run -d --name main" in run and "--memory 2048m" in run and "-w /app" in run
    assert fake.commands[-1] == "sudo -n docker exec -w /app -e 'A=x y' -u root main bash -c 'echo hi'"
    assert fake.deleted == ["bx_1"] and not fake.live
    assert sandbox.usage_seconds == 42.0 and sandbox.sandbox_id is None


def test_failed_start_deletes_the_vm_and_frees_the_slot():
    fake = FakeBoat(fail_on="docker pull")
    quota = BoatQuota(1, 12)
    with pytest.raises(BoatError, match="pull access denied"):
        asyncio.run(_sandbox(fake, quota).start(SPEC))
    assert fake.deleted == ["bx_1"]
    # The single slot is free again: a second start does not wait.
    asyncio.run(asyncio.wait_for(_sandbox(FakeBoat(), quota).start(SPEC), timeout=5))


def test_vm_that_errors_while_provisioning_is_deleted():
    fake = FakeBoat(states=["provisioning", "error"])
    with pytest.raises(BoatError, match="state error"):
        asyncio.run(_sandbox(fake).start(SPEC))
    assert fake.deleted == ["bx_1"]


def test_exec_timeout_stops_the_command():
    fake = FakeBoat(hang_on="sleep 999")
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await sandbox.exec("sleep 999", timeout_s=0)
        finally:
            await sandbox.close()

    asyncio.run(go())
    assert fake.stopped == [len(fake.commands)] and fake.deleted == ["bx_1"]


def test_cancelled_exec_stops_the_command_and_close_still_deletes():
    fake = FakeBoat(hang_on="agent-run")
    sandbox = _sandbox(fake)

    async def go():
        await sandbox.start(SPEC)
        task = asyncio.create_task(sandbox.exec("agent-run"))
        for _ in range(20):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sandbox.close()

    asyncio.run(go())
    assert fake.stopped and fake.deleted == ["bx_1"]


def test_file_transfer_goes_through_staging_and_docker_cp(tmp_path: Path):
    fake = FakeBoat()
    sandbox = _sandbox(fake)
    src = tmp_path / "tests"
    src.mkdir()
    (src / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (src / "test.sh").chmod(0o755)

    async def go():
        await sandbox.start(SPEC)
        await sandbox.upload_bytes(b"payload", "/solution/solve.sh")
        await sandbox.upload_dir(src, "/tests")
        data = await sandbox.download_bytes("/logs/verifier/reward.txt")
        await sandbox.download_dir("/logs/verifier", tmp_path / "out")
        await sandbox.close()
        return data

    data = asyncio.run(go())
    assert data == b"file-from-container"
    assert (tmp_path / "out" / "reward.txt").read_text() == "1\n"
    staged = [p for p in fake.files if p.endswith(".tar.gz") and "/tmp/yeto-xfer/" in p]
    assert staged, "directories travel as one tar archive under /tmp"
    uploaded = tarfile.open(fileobj=io.BytesIO(fake.files[staged[0]]))
    assert uploaded.getmember("./test.sh").mode & 0o111  # exec bit survives
    assert any("docker cp" in c and "main:/solution/solve.sh" in c for c in fake.commands)
    assert any("rm -rf" in c for c in fake.commands)  # staging is cleaned up


def test_pack_and_extract_round_trip(tmp_path: Path):
    (tmp_path / "a" / "empty").mkdir(parents=True)
    (tmp_path / "a" / "f").write_text("x")
    extract_dir(pack_dir(tmp_path / "a"), tmp_path / "b")
    assert (tmp_path / "b" / "f").read_text() == "x" and (tmp_path / "b" / "empty").is_dir()
    with pytest.raises(FileNotFoundError):
        pack_dir(tmp_path / "missing")


def test_docker_exec_quotes_every_argument():
    line = docker_exec_command("echo '$HOME'; rm -rf /", cwd="/a b", env={"K": "v;x"}, user=1000)
    assert line == ("sudo -n docker exec -w '/a b' -e 'K=v;x' -u 1000 main bash -c "
                    "'echo '\"'\"'$HOME'\"'\"'; rm -rf /'")


def test_machine_type_follows_task_resources():
    assert machine_type_for(2048, 1) == "small"
    assert machine_type_for(4096, 1) == "default"
    assert machine_type_for(2048, 4) == "default"
    assert machine_type_for(8192, 2) == "large"


def test_quota_limits_starts_per_minute(monkeypatch):
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(boat_client.asyncio, "sleep", fake_sleep)
    quota = BoatQuota(10, 2, clock=lambda: now[0])

    async def go():
        for _ in range(3):
            await quota.acquire()

    asyncio.run(go())
    assert slept == [60.0]  # the third start waits for the window to slide


def test_quota_active_cap_blocks_until_release():
    quota = BoatQuota(1, 12)

    async def go():
        await quota.acquire()
        waiter = asyncio.create_task(quota.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        quota.release()
        await asyncio.wait_for(waiter, timeout=5)

    asyncio.run(go())
    with pytest.raises(ValueError):
        BoatQuota(0, 12)


def test_in_flight_cap_matches_the_plan(monkeypatch):
    assert check_in_flight_cap(100) == 100
    with pytest.raises(ValueError, match="active-sandbox cap 100"):
        check_in_flight_cap(101)
    monkeypatch.setenv(boat_client.MAX_ACTIVE_ENV, "8")
    with pytest.raises(ValueError):
        check_in_flight_cap(9)


def test_token_comes_from_env_or_cli_config(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(boat_client.API_KEY_ENV, "boat_env_key")
    assert boat_client.load_token() == "boat_env_key"
    monkeypatch.delenv(boat_client.API_KEY_ENV)
    config = tmp_path / "config.json"
    config.write_text('{"token": "cli-token"}')
    monkeypatch.setattr(boat_client, "CLI_CONFIG", config)
    assert boat_client.load_token() == "cli-token"
    config.write_text("{}")
    with pytest.raises(BoatError) as err:
        boat_client.load_token()
    assert "cli-token" not in str(err.value)


# -- Harbor side (needs the pinned Harbor installed) --------------------------


def test_boat_task_maps_to_the_import_path(tmp_path: Path):
    pytest.importorskip("harbor")
    from yeto.rl.agentic import AgentTask
    from yeto.rl.harness.harbor import HarborSettings, build_trial_config
    from yeto.rl.harness.harbor.boat_env import BOAT_IMPORT_PATH

    (tmp_path / "t1").mkdir()
    task = AgentTask.from_mapping({"task_id": "t1", "agent": "oracle", "provider": "boat",
                                   "model_name": "m", "wall_clock_s": 60.0})
    config = build_trial_config(task, "http://x", HarborSettings(tasks_dir=tmp_path))
    assert config.environment.import_path == BOAT_IMPORT_PATH and config.environment.type is None


def test_boat_environment_on_harbor_interface(tmp_path: Path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor.boat_env import BoatEnvironment

    fake = FakeBoat()
    env = BoatEnvironment(
        environment_dir=tmp_path, environment_name="t1", session_id="t1__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="img:1", memory_mb=2048, cpus=1),
        client=fake)
    assert env.type() == "boat" and not env.capabilities.mounted

    async def go():
        await env.start(force_build=False)
        result = await env.exec("true", timeout_sec=10)
        await env.stop(delete=True)
        return result

    result = asyncio.run(go())
    assert result.return_code == 0 and fake.machine_type == "small" and fake.deleted == ["bx_1"]
    with pytest.raises(ValueError, match="docker_image"):
        BoatEnvironment(environment_dir=tmp_path, environment_name="t1", session_id="s",
                        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
                        task_env_config=EnvironmentConfig(), client=fake)
