"""S25 N10.8c: verl's HarborAgentLoop on boat sandboxes, end to end on CPU.

Real pieces: ``HarborAgentLoop.run``, the outer layer, ``build_trial_config``
(provider ``boat`` -> ``import_path``), Harbor's ``Trial`` and the boat
provider.  Fakes: verl's agent-loop module (stub, as in
``test_agentic_harbor``), the boat account (``FakeBoat``) and the token
recorder.  Harbor's ``nop`` agent never calls the model, so the recorder
returns preset turns in place of the calls a policy agent would make.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import test_agentic_harbor as agentic
from test_agentic_harbor import _make_loop, _Recorder, _turn
from test_harbor_boat import FakeBoat

from yeto.rl.agentic import OUTCOME_KEY, HarnessConfigError
from yeto.rl.agentic.loop_local import LoopLocal
from yeto.rl.harness.harbor import boat_client

pytest.importorskip("harbor")

verl_loop = agentic.verl_loop  # the verl agent-loop stub fixture

TASK_TOML = """\
schema_version = "1.1"

[task]
name = "yeto/boat-smoke"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
docker_image = "alpine:3.20"
cpus = 1
memory_mb = 1024
"""


@pytest.fixture
def task_root(tmp_path: Path) -> Path:
    task = tmp_path / "tasks" / "boat-smoke"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    return tmp_path


def _boat_loop(loop_mod, root: Path, fake: FakeBoat, turns, monkeypatch):
    from yeto.rl.harness.harbor import HarborHarness, HarborSettings, boat_env, build_trial_config

    # Harbor writes the trial config to JSON, so the fake account cannot travel as a
    # provider kwarg; it replaces the process client instead.
    monkeypatch.setattr(boat_env, "process_client", lambda: fake)
    settings = HarborSettings(tasks_dir=root / "tasks", trials_dir=root / "trials")
    harness = HarborHarness(lambda task, url: build_trial_config(task, url, settings))
    return _make_loop(loop_mod, _Recorder(turns), harness)


def _row(provider: str | None = "boat") -> dict:
    task = {"task_id": "boat-smoke", "agent": "nop", "trajectory_id": "traj-boat", "wall_clock_s": 120.0}
    if provider is not None:
        task["provider"] = provider
    return {"extra_info": {"agent_task": task}}


@pytest.fixture
def fresh_quota(monkeypatch):
    monkeypatch.setattr(boat_client, "_process_quota", LoopLocal(boat_client.BoatQuota.from_env))


def test_verl_loop_runs_a_harbor_trial_on_a_boat_sandbox(verl_loop, task_root, fresh_quota, monkeypatch):
    loop_mod, _ = verl_loop
    fake = FakeBoat(states=["ready"])
    turns = [_turn([5, 6], [40, 41], [-0.1, -0.2]), _turn([5, 6, 40, 41, 90], [42, 2], [-0.3, -0.4])]
    loop = _boat_loop(loop_mod, task_root, fake, turns, monkeypatch)

    out = asyncio.run(loop.run({"temperature": 1.0}, **_row()))

    assert fake.created == 1 and fake.deleted == ["bx_1"] and not fake.live  # one VM, deleted
    assert any("docker run -d --name main" in c and "alpine:3.20" in c for c in fake.commands)
    assert fake.machine_type == "small"
    assert out.extra_fields[OUTCOME_KEY] == "ok" and out.reward_score == 1.0
    assert out.prompt_ids == [5, 6] and out.response_ids == [40, 41, 90, 42, 2]
    assert out.response_mask == [1, 1, 0, 1, 1]
    assert out.response_logprobs == [-0.1, -0.2, 0.0, -0.3, -0.4]


def test_verl_loop_aborts_when_the_boat_container_fails(verl_loop, task_root, fresh_quota, monkeypatch):
    loop_mod, _ = verl_loop
    fake = FakeBoat(states=["ready"], fail_on="docker pull")
    loop = _boat_loop(loop_mod, task_root, fake, [_turn([5], [6])], monkeypatch)

    async def run_and_snapshot():
        return await loop.run({}, **_row()), loop_mod._gate().metrics.snapshot()

    out, snap = asyncio.run(run_and_snapshot())

    assert fake.deleted == ["bx_1"] and not fake.live  # the failed start still deletes the VM
    assert out.extra_fields[OUTCOME_KEY] == "aborted" and out.reward_score == 0.0
    assert out.response_mask == [0]  # placeholder: no policy-gradient tokens
    assert (snap["harness_outcome_aborted_total"], snap["harness_in_flight"]) == (1, 0)


def test_provider_defaults_to_harbor_env_type_and_boat_cap_is_checked(verl_loop, monkeypatch):
    loop_mod, _ = verl_loop
    monkeypatch.setenv("HARBOR_ENV_TYPE", "Boat")
    assert loop_mod.task_from_row(_row(provider=None)).provider == "boat"
    assert loop_mod.task_from_row(_row(provider="modal")).provider == "modal"  # the row wins

    monkeypatch.setenv("YETO_HARBOR_MAX_IN_FLIGHT", "200")
    with pytest.raises(HarnessConfigError, match="active-sandbox cap"):
        loop_mod.task_from_row(_row())
    loop_mod.task_from_row(_row(provider="modal"))  # the boat cap does not apply to other providers


def test_gate_and_quota_are_per_event_loop(verl_loop, fresh_quota):
    """Two loops in one process (asyncio.run twice) must not share asyncio primitives."""
    loop_mod, _ = verl_loop

    async def hold_and_contend():
        gate, quota = loop_mod._gate(), boat_client.process_quota()
        assert gate is loop_mod._gate() and quota is boat_client.process_quota()  # one per loop
        # Make both primitives wait, which binds them to this loop.
        quota_holders = [asyncio.create_task(quota.acquire()) for _ in range(quota.max_active + 1)]
        await asyncio.sleep(0)
        for task in quota_holders:
            task.cancel()
        await asyncio.gather(*quota_holders, return_exceptions=True)
        return gate, quota

    first = asyncio.run(hold_and_contend())
    second = asyncio.run(hold_and_contend())  # with process singletons: "bound to a different event loop"
    assert first[0] is not second[0] and first[1] is not second[1]
    with pytest.raises(RuntimeError):  # outside a running loop there is no object to hand out
        boat_client.process_quota()
