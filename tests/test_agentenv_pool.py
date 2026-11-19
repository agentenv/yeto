"""CPU tests of AgentENV microVM reuse (pool), exit cleanup, capabilities."""

from __future__ import annotations

import asyncio

import pytest

from yeto.rl.harness.harbor.agentenv_client import AgentenvQuota
from yeto.rl.harness.harbor.agentenv_pool import AgentenvSandboxPool
from yeto.rl.harness.harbor.agentenv_sandbox import AgentenvSandbox, SandboxSpec

from test_harbor_agentenv import FakeAgentenv  # tests/ is on sys.path (rootdir has no __init__)


class VmFake(FakeAgentenv):
    """FakeAgentenv whose ``dirty`` microVMs fail the post-recycle clean check."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dirty: set[str] = set()
        self.owner: dict[int, str] = {}

    def start_command(self, sandbox_id: str, command: str, *, cwd, env, user) -> int:
        process_id = super().start_command(sandbox_id, command, cwd=cwd, env=env, user=user)
        self.owner[process_id] = sandbox_id
        return process_id

    def command_status(self, sandbox_id, process_id):
        command = self.commands[process_id - 1]
        if "echo w=" in command:
            out = "w=1 s=0\n" if sandbox_id in self.dirty else "w=0 s=0\n"
            from yeto.rl.harness.harbor.agentenv_client import CommandStatus
            return CommandStatus(running=False, exit_code=0, stdout=out, stderr="")
        return super().command_status(sandbox_id, process_id)

    def commands_on(self, sandbox_id: str) -> list[str]:
        return [c for i, c in enumerate(self.commands, 1) if self.owner.get(i) == sandbox_id]


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def _factory(*args, **kwargs) -> AgentenvSandbox:
    return AgentenvSandbox(*args, sleep=_no_sleep, poll_interval_s=0.0, **kwargs)


def _pool(fake: VmFake, quota: AgentenvQuota | None = None, clock=None, **kwargs) -> AgentenvSandboxPool:
    extra = {"clock": clock} if clock is not None else {}
    return AgentenvSandboxPool(fake, quota or AgentenvQuota(8, 60), sandbox_factory=_factory, **extra, **kwargs)


SPEC_A = SandboxSpec(template="ubuntu:22.04", workdir="/app")
SPEC_B = SandboxSpec(template="python:3.12", workdir="/app")


def test_pool_reuses_a_recycled_microvm_with_a_fresh_workdir():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        first = await pool.acquire(SPEC_A, name="t1")
        vm = first.sandbox_id
        await pool.release(first)
        second = await pool.acquire(SPEC_A, name="t2")
        assert second is first and second.sandbox_id == vm and second.reuse_count == 1
        assert second.timings["create_s"] == 0.0 and second.name == "t2"
        await pool.release(second)
        await pool.close()
        return pool, vm

    pool, vm = asyncio.run(go())
    assert fake.created == 1 and fake.deleted == [vm] and not fake.live
    on_vm = fake.commands_on(vm)
    recycle = [c for c in on_vm if c.startswith("rm -rf /app")]
    assert len(recycle) == 2  # once per release
    checks = [c for c in on_vm if "echo w=" in c]
    assert len(checks) >= 2  # clean check after each recycle and before each reuse
    assert pool.stats.created == 1 and pool.stats.reused == 1 and pool.stats.idle == 0


def test_pool_keeps_templates_apart():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        a = await pool.acquire(SPEC_A, name="a")
        b = await pool.acquire(SPEC_B, name="b")
        for box in (a, b):
            await pool.release(box)
        again_b = await pool.acquire(SPEC_B, name="d")
        assert again_b is b  # same template, reused
        again_a = await pool.acquire(SPEC_A, name="e")
        assert again_a is a
        await pool.release(again_a)
        await pool.release(again_b)
        await pool.close()

    asyncio.run(go())
    assert fake.created == 2 and not fake.live


def test_microvm_that_fails_the_clean_check_is_deleted_not_reused():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a")
        vm = box.sandbox_id
        fake.dirty.add(vm)
        await pool.release(box)
        assert pool.stats.idle == 0 and pool.stats.discarded == 1
        fresh = await pool.acquire(SPEC_A, name="b")
        assert fresh.sandbox_id != vm
        await pool.release(fresh)
        await pool.close()
        return vm

    vm = asyncio.run(go())
    assert vm in fake.deleted and not fake.live


def test_failed_reuse_falls_back_to_a_new_microvm():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a")
        vm = box.sandbox_id
        await pool.release(box)
        fake.dirty.add(vm)  # dirty while idle: the check before reuse catches it
        fresh = await pool.acquire(SPEC_A, name="b")
        assert fresh.sandbox_id != vm
        await pool.close()
        await fresh.close()
        return vm

    vm = asyncio.run(go())
    assert vm in fake.deleted and not fake.live and fake.created == 2


def test_not_reusable_release_and_reuse_limit_delete_the_microvm():
    fake = VmFake()

    async def go():
        pool = _pool(fake, max_reuses=1)
        box = await pool.acquire(SPEC_A, name="a")
        await pool.release(box, reusable=False)
        assert pool.stats.idle == 0
        box = await pool.acquire(SPEC_A, name="b")
        await pool.release(box)
        box = await pool.acquire(SPEC_A, name="c")  # reuse 1 of 1
        await pool.release(box)
        assert pool.stats.idle == 0
        await pool.close()

    asyncio.run(go())
    assert fake.created == 2 and not fake.live


def test_idle_microvms_expire_and_quota_slots_are_freed():
    fake = VmFake()
    now = [0.0]
    quota = AgentenvQuota(1, 60)

    async def go():
        pool = _pool(fake, quota=quota, clock=lambda: now[0], idle_ttl_s=10.0)
        box = await pool.acquire(SPEC_A, name="a")
        await pool.release(box)
        now[0] = 11.0
        other = await pool.acquire(SPEC_B, name="b")  # the only slot was held by the idle VM
        assert pool.stats.expired == 1
        await other.close()
        await pool.close()

    asyncio.run(go())
    assert fake.created == 2 and not fake.live


def test_idle_reaper_deletes_without_further_calls():
    fake = VmFake()

    async def go():
        pool = _pool(fake, idle_ttl_s=0.05)
        box = await pool.acquire(SPEC_A, name="a")
        await pool.release(box)
        await asyncio.sleep(0.3)
        assert pool.stats.expired == 1 and pool.stats.idle == 0
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_prewarm_creates_idle_microvms_then_serves_acquire():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        assert await pool.prewarm(3, template=SPEC_A.template) == 3
        assert pool.stats.idle == 3
        boxes = await asyncio.gather(*(pool.acquire(SPEC_A, name=f"t{i}") for i in range(3)))
        assert all(b.reuse_count == 1 for b in boxes)
        for box in boxes:
            await pool.release(box)
        await pool.close()

    asyncio.run(go())
    assert fake.created == 3 and not fake.live


def test_fork_path_clones_a_pristine_microvm_off_an_idle_seed():
    fake = VmFake()

    async def go():
        pool = _pool(fake, use_fork=True)
        # First task cold-creates the seed; on release it becomes an idle seed.
        seed_env = await pool.acquire(SPEC_A, name="t1")
        seed_vm = seed_env.sandbox_id
        await pool.release(seed_env)
        assert pool.stats.idle == 1
        # Next tasks fork pristine clones off the seed (seed stays idle).
        clone = await pool.acquire(SPEC_A, name="t2")
        assert clone.forked and clone.sandbox_id != seed_vm and clone.sandbox_id in fake.live
        assert fake.forks == [seed_vm] and pool.stats.idle == 1  # seed untouched
        await pool.release(clone)  # a clone is ephemeral: killed, not pooled
        assert clone.sandbox_id not in fake.live and pool.stats.idle == 1
        assert pool.stats.forked == 1 and pool.stats.created == 1
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_fork_falls_back_to_cold_create_without_a_seed():
    fake = VmFake()

    async def go():
        pool = _pool(fake, use_fork=True)
        box = await pool.acquire(SPEC_A, name="t1")  # no idle seed yet -> cold create
        assert not box.forked and fake.forks == [] and pool.stats.created == 1
        await pool.release(box)
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_closed_pool_deletes_released_microvms():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        box = await pool.acquire(SPEC_A, name="a")
        await pool.close()
        await pool.release(box)
        with pytest.raises(RuntimeError):
            await pool.acquire(SPEC_A, name="b")

    asyncio.run(go())
    assert not fake.live


def test_idle_microvm_gets_a_short_ttl_and_the_task_ttl_back_on_reuse():
    fake = VmFake()

    async def go():
        pool = _pool(fake, idle_vm_ttl_s=1200)
        first = await pool.acquire(SPEC_A, name="t1")
        vm = first.sandbox_id
        await pool.release(first)
        assert fake.ttls == [(vm, 1200)]  # idle in a dead process stops in 20 min, not 3 h
        second = await pool.acquire(SPEC_A, name="t2")
        assert fake.ttls[-1] == (vm, second.ttl_s) and second.ttl_s > 1200
        await pool.close()

    asyncio.run(go())


def test_idle_vm_ttl_must_outlive_the_pool_idle_ttl():
    with pytest.raises(ValueError):
        _pool(VmFake(), idle_ttl_s=300.0, idle_vm_ttl_s=300)


def test_close_now_deletes_idle_and_leased_microvms_without_the_loop():
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        idle = await pool.acquire(SPEC_A, name="t1")
        leased = await pool.acquire(SPEC_B, name="t2")
        await pool.release(idle)
        return pool, {idle.sandbox_id, leased.sandbox_id}

    pool, vms = asyncio.run(go())
    assert pool.close_now() == 2  # e.g. from atexit, after the loop is gone
    assert vms <= set(fake.deleted) and not fake.live
    assert pool.close_now() == 0


def test_shares_boats_exit_cleanup(monkeypatch):
    """register_for_exit / close_pools_now are boat's; an agentenv pool joins them."""
    import weakref

    from yeto.rl.harness.harbor import agentenv_pool, boat_pool

    fake = VmFake()
    monkeypatch.setattr(boat_pool, "_exit_pools", weakref.WeakSet())
    monkeypatch.setattr(boat_pool, "_hooks_installed", False)
    seen = []
    monkeypatch.setattr(boat_pool.atexit, "register", lambda fn: seen.append(fn))

    async def go():
        pool = _pool(fake)
        agentenv_pool.register_for_exit(pool)
        box = await pool.acquire(SPEC_A, name="t1")
        return pool, box.sandbox_id

    pool, vm = asyncio.run(go())  # keep a strong ref: _exit_pools is a WeakSet
    assert boat_pool.close_pools_now() >= 1  # the module-level sweep reaches the agentenv pool
    assert vm in fake.deleted and seen and seen[0] is boat_pool.close_pools_now


# -- Harbor side --------------------------------------------------------------


def _pooled_env(tmp_path, name, pool):
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor.agentenv_env import AgentenvEnvironment

    return AgentenvEnvironment(
        environment_dir=tmp_path, environment_name=name, session_id=f"{name}__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / name),
        task_env_config=EnvironmentConfig(docker_image="img:1", memory_mb=2048, cpus=1), pool=pool)


async def _cancel_while_running(env, command: str) -> None:
    task = asyncio.ensure_future(env.exec(command, timeout_sec=None))
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_pooled_environment_recycles_unless_io_failed(tmp_path):
    pytest.importorskip("harbor")
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        first = _pooled_env(tmp_path, "t1", pool)
        await first.start(force_build=False)
        vm = first.sandbox_id
        await first.stop(delete=True)
        assert pool.stats.idle == 1 and vm in fake.live
        second = _pooled_env(tmp_path, "t2", pool)
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


def test_aborted_trial_returns_its_microvm_through_the_recycle(tmp_path):
    pytest.importorskip("harbor")
    fake = VmFake()

    async def go():
        pool = _pool(fake)
        aborted = _pooled_env(tmp_path, "t1", pool)
        await aborted.start(force_build=False)
        vm = aborted.sandbox_id
        fake.hang_on = "agent-runs-long"
        await _cancel_while_running(aborted, "agent-runs-long")
        fake.hang_on = None
        await aborted.stop(delete=True)
        assert pool.stats.idle == 1 and pool.stats.discarded == 0 and vm in fake.live
        hung = next(i for i, c in enumerate(fake.commands, 1) if "agent-runs-long" in c)
        assert hung in fake.stopped  # the cancelled command was stopped
        nxt = _pooled_env(tmp_path, "t2", pool)
        await nxt.start(force_build=False)
        assert nxt.sandbox_id == vm and fake.created == 1
        await nxt.stop(delete=True)
        await pool.close()

    asyncio.run(go())
    assert not fake.live


def test_pool_is_off_by_default(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from yeto.rl.harness.harbor import agentenv_env

    monkeypatch.delenv("YETO_AGENTENV_POOL", raising=False)
    env = agentenv_env.AgentenvEnvironment(
        environment_dir=tmp_path, environment_name="t", session_id="t__x__env",
        trial_paths=TrialPaths(trial_dir=tmp_path / "t"),
        task_env_config=EnvironmentConfig(docker_image="img:1"), client=VmFake(), quota=AgentenvQuota(1, 30))
    assert env._pool is None
