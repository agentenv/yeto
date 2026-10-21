"""CPU tests of S26 E8: boat VM reuse (pool), start-phase timings, setup retries."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from yeto.rl.harness.harbor.boat_client import BoatQuota, CommandStatus
from yeto.rl.harness.harbor.boat_pool import BoatSandboxPool
from yeto.rl.harness.harbor.boat_sandbox import BoatSandbox, ContainerSpec, _parse_phase_mark
from yeto.rl.harness.harbor.trial import HarborHarness, retryable_setup_failure

from test_harbor_boat import FakeBoat  # pytest puts tests/ on sys.path (rootdir has no __init__)


class VmFake(FakeBoat):
    """FakeBoat whose VMs answer the clean check and the phase line; ``dirty`` VMs fail the check."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dirty: set[str] = set()
        self.owner: dict[int, str] = {}

    def start_command(self, sandbox_id: str, command: str) -> int:
        process_id = super().start_command(sandbox_id, command)
        self.owner[process_id] = sandbox_id
        return process_id

    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus:
        command = self.commands[process_id - 1]
        if "echo c=" in command:
            out = "c=1 v=0 s=0\n" if sandbox_id in self.dirty else "c=0 v=0 s=0\n"
            return CommandStatus(running=False, exit_code=0, stdout=out, stderr="")
        if "yeto-boat-phases" in command and not (self.fail_on and self.fail_on in command):
            return CommandStatus(running=False, exit_code=0, stdout="yeto-boat-phases 2500000000 500000000\n",
                                 stderr="")
        return super().command_status(sandbox_id, process_id)

    def commands_on(self, sandbox_id: str) -> list[str]:
        return [c for i, c in enumerate(self.commands, 1) if self.owner.get(i) == sandbox_id]


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def _factory(*args, **kwargs) -> BoatSandbox:
    return BoatSandbox(*args, sleep=_no_sleep, poll_interval_s=0.0, ready_poll_s=0.0, **kwargs)


def _pool(fake: VmFake, quota: BoatQuota | None = None, clock=None, **kwargs) -> BoatSandboxPool:
    extra = {"clock": clock} if clock is not None else {}
    return BoatSandboxPool(fake, quota or BoatQuota(8, 60), sandbox_factory=_factory, **extra, **kwargs)


SPEC_A = ContainerSpec(image="alexgshaw/fix-git:20251031")
SPEC_B = ContainerSpec(image="alexgshaw/regex-log:20251031")


def test_start_records_phase_timings():
    async def go():
        box = _factory(VmFake(), BoatQuota(2, 12), name="t", machine_type="small")
        await box.start(SPEC_A)
        return box

    box = asyncio.run(go())
    assert set(box.timings) == {"create_s", "ready_s", "container_s", "pull_s", "run_s"}
    assert box.timings["pull_s"] == 2.5 and box.timings["run_s"] == 0.5
    assert box.image == SPEC_A.image


def test_phase_line_parse_tolerates_garbage():
    assert _parse_phase_mark("noise\nyeto-boat-phases 1000000000 2000000000\n") == {"pull_s": 1.0, "run_s": 2.0}
    assert _parse_phase_mark("yeto-boat-phases x y") == {}
    assert _parse_phase_mark("") == {}


def test_pool_reuses_a_recycled_vm_with_a_fresh_container():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        first = await pool.acquire(SPEC_A, name="t1", machine_type="small")
        vm = first.sandbox_id
        await pool.release(first)
        second = await pool.acquire(SPEC_A, name="t2", machine_type="small")
        assert second is first and second.sandbox_id == vm and second.reuse_count == 1
        assert second.timings["create_s"] == 0.0 and second.name == "t2"
        await pool.release(second)
        await pool.close()
        return pool, vm

    pool, vm = asyncio.run(go())
    assert fake.created == 1 and fake.deleted == [vm] and not fake.live
    on_vm = fake.commands_on(vm)
    recycle = [c for c in on_vm if "volume prune" in c]
    assert len(recycle) == 2  # once per release
    assert "rm -f -v" in recycle[0] and "rm -rf /tmp/yeto-xfer" in recycle[0]
    # The second task's container is a new `docker run`, after a clean check.
    runs = [i for i, c in enumerate(on_vm) if "docker run -d --name main" in c]
    checks = [i for i, c in enumerate(on_vm) if "echo c=" in c]
    assert len(runs) == 2 and any(runs[0] < i < runs[1] for i in checks)
    assert pool.stats.created == 1 and pool.stats.reused == 1 and pool.stats.idle == 0


def test_pool_prefers_a_vm_with_the_image_and_keeps_machine_types_apart():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        a = await pool.acquire(SPEC_A, name="a", machine_type="small")
        b = await pool.acquire(SPEC_B, name="b", machine_type="small")
        big = await pool.acquire(SPEC_A, name="c", machine_type="large")
        for box in (a, b, big):
            await pool.release(box)
        again_b = await pool.acquire(SPEC_B, name="d", machine_type="small")
        assert again_b is b
        new_default = await pool.acquire(SPEC_A, name="e", machine_type="default")
        assert new_default not in (a, b, big)
        await pool.release(again_b)
        await pool.release(new_default)
        await pool.close()

    asyncio.run(go())
    assert fake.created == 4 and not fake.live


def test_vm_that_fails_the_clean_check_is_deleted_not_reused():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        vm = box.sandbox_id
        fake.dirty.add(vm)
        await pool.release(box)
        assert pool.stats.idle == 0 and pool.stats.discarded == 1
        fresh = await pool.acquire(SPEC_A, name="b", machine_type="small")
        assert fresh.sandbox_id != vm
        await pool.release(fresh)
        await pool.close()
        return vm

    vm = asyncio.run(go())
    assert vm in fake.deleted and not fake.live


def test_failed_reuse_falls_back_to_a_new_vm():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        vm = box.sandbox_id
        await pool.release(box)
        fake.dirty.add(vm)  # dirty while idle: the check before reuse catches it
        fresh = await pool.acquire(SPEC_A, name="b", machine_type="small")
        assert fresh.sandbox_id != vm
        await pool.close()
        await fresh.close()
        return vm

    vm = asyncio.run(go())
    assert vm in fake.deleted and not fake.live and fake.created == 2


def test_not_reusable_release_and_reuse_limit_delete_the_vm():
    fake = VmFake()

    async def go():
        pool = _pool(fake, max_reuses=1)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        await pool.release(box, reusable=False)
        assert pool.stats.idle == 0
        box = await pool.acquire(SPEC_A, name="b", machine_type="small")
        await pool.release(box)
        box = await pool.acquire(SPEC_A, name="c", machine_type="small")  # reuse 1 of 1
        await pool.release(box)
        assert pool.stats.idle == 0
        await pool.close()

    asyncio.run(go())
    assert fake.created == 2 and not fake.live


def test_idle_vms_expire_and_quota_slots_are_freed():
    fake = VmFake()
    now = [0.0]
    quota = BoatQuota(1, 60)

    async def go():
        pool = _pool(fake, quota=quota, clock=lambda: now[0], idle_ttl_s=10.0)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        await pool.release(box)
        now[0] = 11.0
        # The only quota slot is held by the idle VM until expiry deletes it.
        other = await pool.acquire(SPEC_A, name="b", machine_type="large")
        assert pool.stats.expired == 1
        await other.close()
        await pool.close()

    asyncio.run(go())
    assert fake.created == 2 and not fake.live


def test_idle_reaper_deletes_without_further_calls():
    fake = VmFake()

    async def go():
        pool = _pool(fake, idle_ttl_s=0.05)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        await pool.release(box)
        await asyncio.sleep(0.3)
        assert pool.stats.expired == 1 and pool.stats.idle == 0
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_prewarm_pulls_the_image_without_a_container_then_serves_acquire():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        assert await pool.prewarm(3, machine_type="small", image=SPEC_A.image) == 3
        assert pool.stats.idle == 3
        boxes = await asyncio.gather(*(pool.acquire(SPEC_A, name=f"t{i}", machine_type="small")
                                       for i in range(3)))
        assert all(b.reuse_count == 1 for b in boxes)
        for box in boxes:
            await pool.release(box)
        await pool.close()

    asyncio.run(go())
    assert fake.created == 3 and not fake.live
    assert not any("docker run" in c for c in fake.commands[:6])


def test_failed_prewarm_is_skipped():
    fake = VmFake(fail_on="docker pull")

    async def go():
        pool = _pool(fake)
        started = await pool.prewarm(2, machine_type="small", image=SPEC_A.image)
        await pool.close()
        return started

    assert asyncio.run(go()) == 0
    assert not fake.live


def test_closed_pool_deletes_released_vms():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a", machine_type="small")
        await pool.close()
        await pool.release(box)
        with pytest.raises(RuntimeError):
            await pool.acquire(SPEC_A, name="b", machine_type="small")

    asyncio.run(go())
    assert not fake.live


# -- setup retries -----------------------------------------------------------


def _phase(started: bool = True):
    return SimpleNamespace(started_at=datetime(2026, 10, 21) if started else None, finished_at=None)


def _result(exc: str | None, *, agent_ran: bool = False, rewards=None):
    return SimpleNamespace(
        exception_info=None if exc is None else SimpleNamespace(exception_type=exc, exception_message="boom"),
        verifier_result=None if rewards is None else SimpleNamespace(rewards=rewards),
        agent_execution=_phase() if agent_ran else None,
        environment_setup=None, agent_setup=None, verifier=None, agent_result=None,
        started_at=None, finished_at=None)


def test_retryable_only_before_the_agent_ran():
    assert retryable_setup_failure(_result("NonZeroAgentExitCodeError")) == "NonZeroAgentExitCodeError"
    assert retryable_setup_failure(_result("BoatError")) == "BoatError"
    assert retryable_setup_failure(_result("EnvironmentStartTimeoutError")) == "EnvironmentStartTimeoutError"
    assert retryable_setup_failure(_result("AgentSetupTimeoutError")) == "AgentSetupTimeoutError"
    # After the agent's first command: outcome.classify decides (POLICY_LIMIT or ABORTED).
    assert retryable_setup_failure(_result("NonZeroAgentExitCodeError", agent_ran=True)) is None
    assert retryable_setup_failure(_result("AgentTimeoutError")) is None  # the policy's own budget
    assert retryable_setup_failure(_result("ValueError")) is None  # bad task: same on every sandbox
    assert retryable_setup_failure(_result("BoatQuotaError")) is None  # the account cap stays
    assert retryable_setup_failure(_result(None, rewards={"reward": 1.0})) is None


class _ScriptedTrial:
    script: list = []
    built = 0

    def __init__(self, result):
        self._result = result
        self.paths = None

    @classmethod
    async def create(cls, config):
        cls.built += 1
        return cls(cls.script.pop(0))

    async def run(self):
        return self._result


def _harness(script, retries):
    _ScriptedTrial.script = list(script)
    _ScriptedTrial.built = 0
    delays: list[float] = []

    async def sleep(seconds):
        delays.append(seconds)

    harness = HarborHarness(lambda task, url: {}, trial_cls=_ScriptedTrial, setup_retries=retries,
                            retry_backoff_s=4.0, sleep=sleep)
    return harness, delays


TASK = SimpleNamespace(task_id="fix-git")


def test_setup_failure_is_retried_then_succeeds():
    harness, delays = _harness([_result("NonZeroAgentExitCodeError"), _result("BoatError"),
                                _result(None, rewards={"reward": 1.0})], retries=3)
    verdict = asyncio.run(harness.run(TASK, "http://e"))
    assert verdict.exception_type is None and verdict.rewards == {"reward": 1.0}
    assert verdict.agent_metrics["setup_attempts"] == 3
    assert verdict.agent_metrics["setup_retried"] == ["NonZeroAgentExitCodeError", "BoatError"]
    assert 2.0 <= delays[0] <= 4.0 and 4.0 <= delays[1] <= 8.0  # exponential with jitter


def test_retries_spent_keep_the_infra_exception_for_aborted():
    harness, _ = _harness([_result("BoatError")] * 3, retries=2)
    verdict = asyncio.run(harness.run(TASK, "http://e"))
    assert verdict.exception_type == "BoatError" and verdict.agent_metrics["setup_attempts"] == 3
    assert _ScriptedTrial.built == 3


def test_failure_after_the_agent_ran_is_not_retried():
    harness, delays = _harness([_result("NonZeroAgentExitCodeError", agent_ran=True)], retries=3)
    verdict = asyncio.run(harness.run(TASK, "http://e"))
    assert verdict.exception_type == "NonZeroAgentExitCodeError" and not delays
    assert verdict.agent_metrics["setup_attempts"] == 1


def test_default_harness_does_not_retry_and_adds_no_metrics():
    _ScriptedTrial.script = [_result("BoatError")]
    _ScriptedTrial.built = 0
    verdict = asyncio.run(HarborHarness(lambda t, u: {}, trial_cls=_ScriptedTrial).run(TASK, "http://e"))
    assert verdict.exception_type == "BoatError" and _ScriptedTrial.built == 1
    assert "setup_attempts" not in verdict.agent_metrics


def test_setup_retries_env(monkeypatch):
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.harness.harbor.trial import setup_retries_from_env

    monkeypatch.delenv("HARBOR_SETUP_RETRIES", raising=False)
    assert setup_retries_from_env() == 0
    monkeypatch.setenv("HARBOR_SETUP_RETRIES", "2")
    assert setup_retries_from_env() == 2
    for bad in ("-1", "x"):
        monkeypatch.setenv("HARBOR_SETUP_RETRIES", bad)
        with pytest.raises(HarnessConfigError):
            setup_retries_from_env()


def test_pooled_boat_environment_recycles_unless_io_failed(tmp_path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor.boat_env import BoatEnvironment

    fake = VmFake()

    def env(name: str, pool: BoatSandboxPool) -> BoatEnvironment:
        return BoatEnvironment(
            environment_dir=tmp_path, environment_name=name, session_id=f"{name}__x__env",
            trial_paths=TrialPaths(trial_dir=tmp_path / name),
            task_env_config=EnvironmentConfig(docker_image="img:1", memory_mb=2048, cpus=1), pool=pool)

    async def go():
        pool = _pool(fake)
        first = env("t1", pool)
        await first.start(force_build=False)
        vm = first.sandbox_id
        await first.stop(delete=True)
        assert pool.stats.idle == 1 and vm in fake.live
        second = env("t2", pool)
        await second.start(force_build=False)
        assert second.sandbox_id == vm and second.start_timings["create_s"] == 0.0
        fake.hang_on = "never-ends"
        with pytest.raises(RuntimeError, match="timed out"):
            await second.exec("never-ends", timeout_sec=0)
        await second.stop(delete=True)  # an exec failed on it: deleted, not pooled
        assert pool.stats.idle == 0 and vm not in fake.live
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_pool_is_off_by_default(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor import boat_env

    monkeypatch.delenv("YETO_BOAT_POOL", raising=False)
    env = boat_env.BoatEnvironment(
        environment_dir=tmp_path, environment_name="t", session_id="t__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "t"),
        task_env_config=EnvironmentConfig(docker_image="img:1"), client=VmFake(), quota=BoatQuota(1, 12))
    assert env._pool is None


def test_quota_hourly_window(monkeypatch):
    from yeto.rl.harness.harbor import boat_client

    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(boat_client.asyncio, "sleep", fake_sleep)
    quota = BoatQuota(100, 12, starts_per_hour=3, clock=lambda: now[0])

    async def go():
        for _ in range(4):
            await quota.acquire()

    asyncio.run(go())
    assert slept == [3600.0]  # the fourth start waits for the first to leave the hour


def test_quota_hourly_from_env(monkeypatch):
    monkeypatch.delenv("YETO_BOAT_STARTS_PER_HOUR", raising=False)
    assert BoatQuota.from_env().starts_per_hour is None  # default: unchanged behaviour
    monkeypatch.setenv("YETO_BOAT_STARTS_PER_HOUR", "60")
    assert BoatQuota.from_env().starts_per_hour == 60
