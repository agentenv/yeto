from __future__ import annotations

import asyncio
import copy
import json
import sys
import types
from io import BytesIO
from types import SimpleNamespace

import pytest

from yeto_miles_secrlenv import agent, reward
from yeto_miles_secrlenv.client import (
    EpisodeAPIError,
    EpisodeClient,
    EpisodeClientError,
    EpisodeTransportError,
    require_daemon_ready,
)


class FakeMessage:
    def __init__(self, value):
        self.value = value

    def model_dump(self, *, exclude_none=True):
        return copy.deepcopy(self.value)


class FakeCompletions:
    def __init__(self, messages):
        self.responses = list(messages)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FakeMessage(self.responses.pop(0)))]
        )


class FakePolicy:
    def __init__(self, messages):
        self.chat = SimpleNamespace(completions=FakeCompletions(messages))


class FakeEpisodeClient:
    def __init__(self):
        self.exec_calls = []
        self.submissions = []

    async def execute(
        self, episode_id, command, *, timeout_seconds, output_bytes
    ):
        self.exec_calls.append(
            (episode_id, command, timeout_seconds, output_bytes)
        )
        return {
            "exit_code": 0,
            "output": "command output",
            "timed_out": False,
            "truncated": False,
        }

    async def submit(self, episode_id, submission):
        self.submissions.append((episode_id, submission))
        return {"accepted": True}


def _install_run_openai(monkeypatch):
    class RunPolicy:
        async def close(self):
            return None

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = lambda **_kwargs: RunPolicy()
    monkeypatch.setitem(sys.modules, "openai", openai)


def _run_metadata():
    return {"task_id": "CVE-2024-1234", "prompt_tier": "l2"}


def test_episode_finalization_cannot_be_claimed_by_abort():
    episode_id = "f" * 24
    agent._register_episode(episode_id)
    try:
        assert agent._claim_episode_finalization(episode_id)
        assert agent._claim_driving_episodes_for_abort() == []
    finally:
        agent._release_episode(episode_id)


def test_abort_claim_preempts_finalization_once():
    episode_id = "a" * 24
    agent._register_episode(episode_id)
    try:
        assert agent._claim_driving_episodes_for_abort() == [episode_id]
        assert not agent._claim_episode_finalization(episode_id)
        assert not agent._claim_episode_cleanup(episode_id)
    finally:
        agent._release_episode(episode_id)


def test_finalizing_episode_cannot_be_claimed_for_cancelled_cleanup():
    episode_id = "e" * 24
    agent._register_episode(episode_id)
    try:
        assert agent._claim_episode_finalization(episode_id)
        assert not agent._claim_episode_cleanup(episode_id)
        assert agent._claim_episode_finalization(episode_id)
    finally:
        agent._release_episode(episode_id)


def test_policy_completion_claims_finalization_before_abort_callback():
    episode_id = "b" * 24
    abort_claims = []

    async def complete_policy():
        asyncio.get_running_loop().call_soon(
            lambda: abort_claims.extend(
                agent._claim_driving_episodes_for_abort()
            )
        )
        return "completed"

    async def scenario():
        result = await agent._await_policy_and_claim_finalization(
            episode_id, complete_policy()
        )
        await asyncio.sleep(0)
        return result

    agent._register_episode(episode_id)
    try:
        assert asyncio.run(scenario()) == "completed"
        assert abort_claims == []
        assert agent._claim_episode_finalization(episode_id)
    finally:
        agent._release_episode(episode_id)


def test_registered_policy_task_includes_finalization_boundary():
    episode_id = "6" * 24
    abort_claims = []

    async def complete_policy():
        asyncio.get_running_loop().call_soon(
            lambda: abort_claims.extend(
                agent._claim_driving_episodes_for_abort()
            )
        )
        return "completed"

    async def scenario():
        policy_task = asyncio.create_task(
            agent._await_policy_and_claim_finalization(
                episode_id, complete_policy()
            )
        )
        agent._register_episode(episode_id, policy_task)
        result = await asyncio.wait_for(policy_task, timeout=1.0)
        await asyncio.sleep(0)
        return result

    try:
        assert asyncio.run(scenario()) == "completed"
        assert abort_claims == []
        assert agent._claim_episode_finalization(episode_id)
    finally:
        agent._release_episode(episode_id)


def test_cancelled_run_can_transfer_its_drained_boundary_to_cleanup():
    episode_id = "5" * 24

    async def scenario():
        policy_task = asyncio.create_task(
            agent._await_policy_and_claim_finalization(
                episode_id, asyncio.sleep(0, result="completed")
            )
        )
        agent._register_episode(episode_id, policy_task)
        await policy_task
        assert not agent._claim_episode_cleanup(episode_id)
        assert agent._claim_cancelled_policy_cleanup(episode_id, policy_task)

    try:
        asyncio.run(scenario())
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
    finally:
        agent._release_episode(episode_id)


def test_run_cancellation_drains_evaluation_before_cleanup_close(monkeypatch):
    episode_id = "3" * 24
    evaluation_started = None
    evaluation_live = False
    close_observations = []

    class RunClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def create(self, _task_id, _tier):
            return {"episode_id": episode_id, "prompt": "task"}

        async def evaluate(self, value):
            assert value == episode_id
            nonlocal evaluation_live
            evaluation_live = True
            evaluation_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                evaluation_live = False

        async def close(self, value):
            assert value == episode_id
            close_observations.append(evaluation_live)
            return {}

    async def complete_policy(*_args, **_kwargs):
        return "completed"

    async def scenario():
        nonlocal evaluation_started
        evaluation_started = asyncio.Event()
        run_task = asyncio.create_task(
            agent.run("http://127.0.0.1:8000/v1", None, metadata=_run_metadata())
        )
        await evaluation_started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    _install_run_openai(monkeypatch)
    monkeypatch.setattr(agent, "EpisodeClient", RunClient)
    monkeypatch.setattr(agent, "_drive_policy", complete_policy)
    asyncio.run(scenario())
    assert close_observations == [False]
    assert episode_id not in agent._EPISODE_PHASES


def test_run_cancellation_drains_normal_close_before_cleanup_retry(monkeypatch):
    episode_id = "2" * 24
    first_close_started = None
    first_close_live = False
    close_observations = []

    class RunClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def create(self, _task_id, _tier):
            return {"episode_id": episode_id, "prompt": "task"}

        async def evaluate(self, value):
            assert value == episode_id
            return {}

        async def close(self, value):
            assert value == episode_id
            nonlocal first_close_live
            close_observations.append(first_close_live)
            if len(close_observations) == 1:
                first_close_live = True
                first_close_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    first_close_live = False
            assert not first_close_live
            return {}

    async def complete_policy(*_args, **_kwargs):
        return "completed"

    async def scenario():
        nonlocal first_close_started
        first_close_started = asyncio.Event()
        run_task = asyncio.create_task(
            agent.run("http://127.0.0.1:8000/v1", None, metadata=_run_metadata())
        )
        await first_close_started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    _install_run_openai(monkeypatch)
    monkeypatch.setattr(agent, "EpisodeClient", RunClient)
    monkeypatch.setattr(agent, "_drive_policy", complete_policy)
    asyncio.run(scenario())
    assert close_observations == [False, False]
    assert episode_id not in agent._EPISODE_PHASES


def test_run_close_failure_retains_cleanup_pending_ownership(monkeypatch):
    episode_id = "1" * 24

    class RunClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def create(self, _task_id, _tier):
            return {"episode_id": episode_id, "prompt": "task"}

        async def evaluate(self, value):
            assert value == episode_id
            return {
                "task_id": "CVE-2024-1234",
                "episode_id": episode_id,
                "reward": 0.0,
                "passed": False,
            }

        async def close(self, value):
            assert value == episode_id
            raise EpisodeTransportError("close failed")

    async def complete_policy(*_args, **_kwargs):
        return "completed"

    _install_run_openai(monkeypatch)
    monkeypatch.setattr(agent, "EpisodeClient", RunClient)
    monkeypatch.setattr(agent, "_drive_policy", complete_policy)
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "k" * 48)
    try:
        result = asyncio.run(
            agent.run("http://127.0.0.1:8000/v1", None, metadata=_run_metadata())
        )
        assert result is not None
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
        assert agent._claim_episodes_and_tasks_for_abort() == [
            (episode_id, agent._EPISODE_POLICY_TASKS[episode_id])
        ]
    finally:
        agent._release_episode(episode_id)


def test_second_run_cancellation_drains_cleanup_close_before_handoff(monkeypatch):
    episode_id = "0" * 24
    policy_started = None
    cleanup_close_started = None
    close_live = False
    close_observations = []

    class RunClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def create(self, _task_id, _tier):
            return {"episode_id": episode_id, "prompt": "task"}

        async def close(self, value):
            assert value == episode_id
            nonlocal close_live
            close_observations.append(close_live)
            if len(close_observations) == 1:
                close_live = True
                cleanup_close_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    close_live = False
            assert not close_live
            return {}

    async def blocked_policy(*_args, **_kwargs):
        policy_started.set()
        await asyncio.Event().wait()

    async def scenario():
        nonlocal policy_started, cleanup_close_started
        policy_started = asyncio.Event()
        cleanup_close_started = asyncio.Event()
        run_task = asyncio.create_task(
            agent.run("http://127.0.0.1:8000/v1", None, metadata=_run_metadata())
        )
        await policy_started.wait()
        run_task.cancel()
        await cleanup_close_started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert not close_live
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
        await agent.abort()

    _install_run_openai(monkeypatch)
    monkeypatch.setattr(agent, "EpisodeClient", RunClient)
    monkeypatch.setattr(agent, "_drive_policy", blocked_policy)
    asyncio.run(scenario())
    assert close_observations == [False, False]
    assert episode_id not in agent._EPISODE_PHASES


def test_abort_drains_pending_policy_before_close(monkeypatch):
    episode_id = "d" * 24
    events = []

    class AbortClient:
        def __init__(self, *, total_timeout_seconds):
            assert total_timeout_seconds == 180.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def close(self, value):
            assert value == episode_id
            assert policy_task.done()
            events.append("close")
            return {}

    async def scenario():
        started = asyncio.Event()

        async def pending_policy():
            events.append("policy_started")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("policy_drained")

        nonlocal policy_task
        policy_task = asyncio.create_task(pending_policy())
        agent._register_episode(episode_id, policy_task)
        await started.wait()
        await agent.abort()
        assert policy_task.cancelled()

    policy_task = None
    monkeypatch.setattr(agent, "EpisodeClient", AbortClient)
    try:
        asyncio.run(scenario())
        assert events == ["policy_started", "policy_drained", "close"]
        assert episode_id not in agent._EPISODE_PHASES
    finally:
        agent._release_episode(episode_id)


def test_abort_close_retries_episode_conflict_without_releasing_ownership(
    monkeypatch,
):
    episode_id = "9" * 24
    phases = []

    class ConflictClient:
        async def close(self, value):
            assert value == episode_id
            phases.append(agent._EPISODE_PHASES.get(episode_id))
            if len(phases) < 3:
                raise EpisodeAPIError(409, "episode_conflict", "still active")
            return {}

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(agent.asyncio, "sleep", no_wait)
    agent._register_episode(episode_id)
    try:
        assert agent._claim_driving_episodes_for_abort() == [episode_id]
        assert asyncio.run(
            agent._close_aborted_episode(
                ConflictClient(),
                episode_id,
                retry_timeout_seconds=1.0,
            )
        )
        assert phases == ["aborting", "aborting", "aborting"]
        assert episode_id not in agent._EPISODE_PHASES
    finally:
        agent._release_episode(episode_id)


@pytest.mark.parametrize(
    ("status", "code"),
    [(409, "episode_conflict"), (500, "internal_error")],
)
def test_failed_abort_close_retains_retryable_cleanup_ownership(status, code):
    episode_id = "8" * 24

    class FailingClient:
        async def close(self, value):
            assert value == episode_id
            raise EpisodeAPIError(status, code, "not closed")

    agent._register_episode(episode_id)
    try:
        assert agent._claim_driving_episodes_for_abort() == [episode_id]
        assert not asyncio.run(
            agent._close_aborted_episode(
                FailingClient(),
                episode_id,
                retry_timeout_seconds=0.0,
            )
        )
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
        assert agent._claim_episodes_and_tasks_for_abort() == []
        agent._mark_episode_cleanup_pending(episode_id)
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
        assert agent._claim_episodes_and_tasks_for_abort() == [(episode_id, None)]
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
    finally:
        agent._release_episode(episode_id)


def test_failed_close_cannot_be_reclaimed_before_owner_finishes(monkeypatch):
    episode_id = "4" * 24

    class FailingClient:
        async def close(self, value):
            assert value == episode_id
            raise EpisodeAPIError(500, "internal_error", "not closed")

    class UnexpectedAbortClient:
        def __init__(self, **_kwargs):
            raise AssertionError("overlapping abort reclaimed live cleanup ownership")

    async def scenario():
        assert not await agent._close_aborted_episode(
            FailingClient(),
            episode_id,
            retry_timeout_seconds=0.0,
        )
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
        await agent.abort()
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
        agent._mark_episode_cleanup_pending(episode_id)

    monkeypatch.setattr(agent, "EpisodeClient", UnexpectedAbortClient)
    agent._register_episode(episode_id)
    try:
        assert agent._claim_driving_episodes_for_abort() == [episode_id]
        asyncio.run(scenario())
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
    finally:
        agent._release_episode(episode_id)


def test_timed_out_policy_drain_retains_live_owner_until_abort_returns():
    episode_id = "a" * 23 + "0"

    async def scenario():
        release = asyncio.Event()

        async def slow_cancellation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(slow_cancellation())
        agent._register_episode(episode_id, task)
        await asyncio.sleep(0)
        assert agent._claim_episodes_and_tasks_for_abort() == [(episode_id, task)]
        assert not await agent._drain_aborted_policy_task(task, timeout_seconds=0.0)
        assert agent._EPISODE_PHASES[episode_id] == "aborting"
        assert agent._claim_episodes_and_tasks_for_abort() == []
        release.set()
        await task

    try:
        asyncio.run(scenario())
    finally:
        agent._release_episode(episode_id)


def test_abort_without_claimable_episodes_does_not_initialize_client(monkeypatch):
    class UnexpectedClient:
        def __init__(self, **_kwargs):
            raise AssertionError("abort client initialized without cleanup work")

    monkeypatch.setattr(agent, "EpisodeClient", UnexpectedClient)
    asyncio.run(agent.abort())


def test_cancelled_abort_retains_cleanup_pending_ownership(monkeypatch):
    episode_id = "7" * 24

    class BlockingClient:
        def __init__(self, *, total_timeout_seconds):
            assert total_timeout_seconds == 180.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def close(self, value):
            assert value == episode_id
            close_started.set()
            await asyncio.Event().wait()

    async def scenario():
        nonlocal close_started
        close_started = asyncio.Event()
        abort_task = asyncio.create_task(agent.abort())
        await close_started.wait()
        abort_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abort_task

    close_started = None
    monkeypatch.setattr(agent, "EpisodeClient", BlockingClient)
    agent._register_episode(episode_id)
    try:
        asyncio.run(scenario())
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
        assert agent._claim_episodes_and_tasks_for_abort() == [(episode_id, None)]
    finally:
        agent._release_episode(episode_id)


def test_abort_client_initialization_failure_retains_cleanup_ownership(monkeypatch):
    episode_id = "c" * 24

    class FailingEpisodeClient:
        def __init__(self, *, total_timeout_seconds):
            assert total_timeout_seconds == 180.0

        async def __aenter__(self):
            raise RuntimeError("client unavailable")

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    monkeypatch.setattr(agent, "EpisodeClient", FailingEpisodeClient)
    agent._register_episode(episode_id)
    try:
        asyncio.run(agent.abort())
        assert not agent._claim_episode_finalization(episode_id)
        assert agent._EPISODE_PHASES[episode_id] == "cleanup_pending"
        assert agent._claim_episodes_and_tasks_for_abort() == [(episode_id, None)]
    finally:
        agent._release_episode(episode_id)


def _tool_call(identifier, name, arguments):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_adapter_is_network_only_and_enforces_l2():
    prompt_surface = agent.SYSTEM_PROMPT + json.dumps(agent.TOOLS)
    assert "DOCKER_HOST" not in prompt_surface
    assert "DEBUG_URL" not in prompt_surface
    assert "debug daemon" not in prompt_surface
    assert agent._task_identity({"task_id": "CVE-2024-1234"}) == (
        "CVE-2024-1234",
        "l2",
    )
    assert agent._task_identity(
        {"task_id": "CVE-2024-1234", "prompt_tier": "l2"}
    ) == ("CVE-2024-1234", "l2")
    for tier in ("l0", "l1"):
        with pytest.raises(ValueError, match="must use prompt_tier l2"):
            agent._task_identity(
                {"task_id": "CVE-2024-1234", "prompt_tier": tier}
            )


def test_agent_preserves_full_assistant_message_and_uses_tool_roles(monkeypatch):
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "4")
    policy = FakePolicy(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "inspect first",
                "tool_calls": [
                    _tool_call("call-1", "terminal.exec", {"command": "id"})
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "submit evidence",
                "tool_calls": [
                    _tool_call(
                        "call-2", "submit", {"evidence": "target returned nonce"}
                    )
                ],
            },
        ]
    )
    client = FakeEpisodeClient()
    metrics = agent.AgentMetrics()
    status = asyncio.run(
        agent._drive_policy(
            policy,
            client,
            {"episode_id": "a" * 24, "prompt": "solve this"},
            {"temperature": 0.7, "stream": True},
            metrics,
        )
    )
    assert status == "completed"
    assert client.exec_calls[0][1] == "id"
    assert client.submissions == [
        ("a" * 24, {"evidence": "target returned nonce"})
    ]
    second_messages = policy.chat.completions.requests[1]["messages"]
    first_assistant = second_messages[2]
    assert first_assistant["reasoning_content"] == "inspect first"
    assert first_assistant["tool_calls"][0]["id"] == "call-1"
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": json.dumps(
            {
                "exit_code": 0,
                "output": "command output",
                "timed_out": False,
                "truncated": False,
            },
            separators=(",", ":"),
        ),
    }
    assert "stream" not in policy.chat.completions.requests[0]["extra_body"]
    assert metrics.terminal_calls == 1
    assert metrics.submit_calls == 1


def test_agent_retries_non_tool_prose_as_user_turn(monkeypatch):
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "3")
    monkeypatch.setenv("SECRLENV_MAX_NO_TOOL_RETRIES", "2")
    policy = FakePolicy(
        [
            {"role": "assistant", "content": "I would run id"},
            {"role": "assistant", "content": "Still no tool"},
        ]
    )
    metrics = agent.AgentMetrics()
    status = asyncio.run(
        agent._drive_policy(
            policy,
            FakeEpisodeClient(),
            {"episode_id": "b" * 24, "prompt": "solve"},
            {},
            metrics,
        )
    )
    assert status == "max_turns"
    assert policy.chat.completions.requests[1]["messages"][-1]["role"] == "user"
    assert metrics.parse_failures == 2


def test_signed_zero_reward_is_valid_and_tampering_is_rejected(monkeypatch):
    key = "k" * 48
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", key)
    outcome = {
        "schema": 1,
        "status": "completed",
        "episode_id": "c" * 24,
        "task_id": "CVE-2024-1234",
        "reward": 0.0,
        "passed": False,
        "class": "rce",
    }
    metadata = {
        reward.OUTCOME_KEY: outcome,
        reward.MAC_KEY: reward.sign_outcome(outcome),
    }
    assert reward.verify_outcome(metadata) == 0.0
    sample = SimpleNamespace(metadata=metadata)
    assert asyncio.run(reward.reward_func(None, sample)) == 0.0

    forged = copy.deepcopy(metadata)
    forged[reward.OUTCOME_KEY]["reward"] = 1.0
    with pytest.raises(reward.UntrustedOutcome, match="signature mismatch"):
        reward.verify_outcome(forged)


def test_sequence_cap_outcome_contract_and_unknown_status(monkeypatch):
    key = "s" * 48
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", key)
    task_id = "CVE-2024-1234"
    episode_id = "m" * 24
    outcome = agent._validated_outcome(
        {
            "task_id": task_id,
            "episode_id": episode_id,
            "reward": 0.25,
            "passed": False,
            "grader": {"class": "rce"},
        },
        task_id,
        episode_id,
        "max_seq_len",
    )
    metadata = {
        reward.OUTCOME_KEY: outcome,
        reward.MAC_KEY: reward.sign_outcome(outcome),
    }

    assert reward.verify_outcome(metadata) == 0.25

    unknown = copy.deepcopy(outcome)
    unknown["status"] = "unknown_termination"
    unknown_metadata = {
        reward.OUTCOME_KEY: unknown,
        reward.MAC_KEY: reward.sign_outcome(unknown),
    }
    with pytest.raises(reward.UntrustedOutcome, match="invalid schema or status"):
        reward.verify_outcome(unknown_metadata)


def test_missing_reward_never_defaults_to_zero(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "z" * 48)
    sample = SimpleNamespace(metadata={"reward": 0.0})
    with pytest.raises(reward.UntrustedOutcome, match="no signed"):
        asyncio.run(reward.reward_func(None, sample))


def _install_filter_types(monkeypatch):
    class DynamicFilterOutput:
        def __init__(self, *, keep, reason):
            self.keep = keep
            self.reason = reason

    class Sample:
        class Status:
            ABORTED = "aborted"

    miles = types.ModuleType("miles")
    rollout = types.ModuleType("miles.rollout")
    filter_hub = types.ModuleType("miles.rollout.filter_hub")
    base_types = types.ModuleType("miles.rollout.filter_hub.base_types")
    utils = types.ModuleType("miles.utils")
    sample_types = types.ModuleType("miles.utils.types")
    base_types.DynamicFilterOutput = DynamicFilterOutput
    sample_types.Sample = Sample
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.filter_hub", filter_hub)
    monkeypatch.setitem(
        sys.modules, "miles.rollout.filter_hub.base_types", base_types
    )
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.types", sample_types)


def _signed_sample(index, value, *, status="completed"):
    outcome = {
        "schema": 1,
        "status": "completed",
        "episode_id": f"episode-{index:024d}",
        "task_id": "CVE-2019-7859",
        "reward": value,
    }
    return SimpleNamespace(
        index=index,
        status=status,
        metadata={
            reward.OUTCOME_KEY: outcome,
            reward.MAC_KEY: reward.sign_outcome(outcome),
        },
    )


def test_signed_infrastructure_outcome_is_only_an_abort_signal(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "i" * 48)
    _install_filter_types(monkeypatch)
    outcome = agent._infrastructure_outcome(
        "CVE-2024-1234", "f" * 24
    )
    metadata = {
        reward.OUTCOME_KEY: outcome,
        reward.MAC_KEY: reward.sign_outcome(outcome),
    }
    sample = SimpleNamespace(index=30, status="completed", metadata=metadata)

    assert asyncio.run(reward.reward_func(None, sample)) == 0.0
    assert sample.status == "aborted"
    with pytest.raises(
        reward.UntrustedOutcome, match="not a grader verdict"
    ):
        reward.verify_outcome(metadata)

    result = reward.check_group(
        SimpleNamespace(
            yeto_rl_policy_version=10,
            yeto_rl_dynamic_sampling_max_replacements=0,
        ),
        [sample],
    )
    assert (result.keep, result.reason) == (False, "secrlenv_aborted")

    direct_sample = SimpleNamespace(
        index=31, status="completed", metadata=copy.deepcopy(metadata)
    )
    direct = reward.check_group(
        SimpleNamespace(
            yeto_rl_policy_version=10,
            yeto_rl_dynamic_sampling_max_replacements=0,
        ),
        [direct_sample],
    )
    assert (direct.keep, direct.reason) == (
        False,
        "secrlenv_infrastructure_failure",
    )

    forged = copy.deepcopy(metadata)
    forged[reward.OUTCOME_KEY]["task_id"] = "CVE-2024-9999"
    forged_sample = SimpleNamespace(
        index=32, status="completed", metadata=forged
    )
    with pytest.raises(reward.UntrustedOutcome, match="signature mismatch"):
        asyncio.run(reward.reward_func(None, forged_sample))
    assert forged_sample.status == "completed"


def test_signed_group_filter_accepts_valid_reward_variance(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "v" * 48)
    _install_filter_types(monkeypatch)
    args = SimpleNamespace(
        yeto_rl_policy_version=7,
        yeto_rl_dynamic_sampling_max_replacements=2,
    )

    result = reward.check_group(
        args, [_signed_sample(1, 0.0), _signed_sample(2, 1.0)]
    )

    assert result.keep is True
    assert result.reason is None
    assert args._yeto_secrlenv_filter_state["rejections"] == 0


def test_signed_zero_variance_groups_have_a_memoized_replacement_bound(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "b" * 48)
    _install_filter_types(monkeypatch)
    args = SimpleNamespace(
        yeto_rl_policy_version=8,
        yeto_rl_dynamic_sampling_max_replacements=2,
    )
    first = [_signed_sample(10, 0.0), _signed_sample(11, 0.0)]
    second = [_signed_sample(12, 1.0), _signed_sample(13, 1.0)]
    fallback = [_signed_sample(14, 0.0), _signed_sample(15, 0.0)]

    decisions = [
        reward.check_group(args, first),
        reward.check_group(args, second),
        reward.check_group(args, fallback),
        reward.check_group(args, fallback),
    ]

    assert [decision.keep for decision in decisions] == [False, False, True, True]
    assert decisions[2].reason == "secrlenv_bounded_fallback_after_2_replacements"
    assert args._yeto_secrlenv_filter_state["rejections"] == 2
    assert args._yeto_secrlenv_filter_state["forced"] == 1


def test_aborted_and_untrusted_groups_never_use_bounded_fallback(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "u" * 48)
    _install_filter_types(monkeypatch)
    args = SimpleNamespace(
        yeto_rl_policy_version=9,
        yeto_rl_dynamic_sampling_max_replacements=0,
    )
    aborted = [_signed_sample(20, 0.0, status="aborted")]
    forged = _signed_sample(21, 0.0)
    forged.metadata[reward.OUTCOME_KEY]["reward"] = 1.0

    aborted_result = reward.check_group(args, aborted)
    forged_result = reward.check_group(args, [forged])

    assert (aborted_result.keep, aborted_result.reason) == (
        False,
        "secrlenv_aborted",
    )
    assert (forged_result.keep, forged_result.reason) == (
        False,
        "secrlenv_untrusted_outcome",
    )
    assert not hasattr(args, "_yeto_secrlenv_filter_state")


def test_evaluation_identity_and_range_are_checked():
    with pytest.raises(EpisodeClientError, match="identity"):
        agent._validated_outcome(
            {
                "task_id": "CVE-2024-9999",
                "episode_id": "d" * 24,
                "reward": 0.5,
                "passed": False,
            },
            "CVE-2024-1234",
            "d" * 24,
            "completed",
        )
    with pytest.raises(EpisodeClientError, match="invalid verdict"):
        agent._validated_outcome(
            {
                "task_id": "CVE-2024-1234",
                "episode_id": "d" * 24,
                "reward": 2.0,
                "passed": False,
            },
            "CVE-2024-1234",
            "d" * 24,
            "completed",
        )


def test_client_refuses_non_loopback_or_credentialed_origins():
    token = "t" * 48
    for value in (
        "http://example.com:8765",
        "https://127.0.0.1:8765",
        "http://user:pass@127.0.0.1:8765",
        "http://127.0.0.1:8765/path",
    ):
        with pytest.raises(EpisodeClientError):
            EpisodeClient(value, token)


def test_daemon_readiness_fails_fast_without_a_listener(monkeypatch):
    monkeypatch.setenv("SECRLENV_DAEMON_URL", "http://127.0.0.1:8765")

    def unavailable(*_args, **_kwargs):
        raise OSError

    monkeypatch.setattr("yeto_miles_secrlenv.client.urlopen", unavailable)
    with pytest.raises(EpisodeTransportError, match="not healthy"):
        require_daemon_ready(timeout_seconds=0.1)


def test_daemon_readiness_attests_health_and_task_pack(monkeypatch):
    expected = "a" * 64
    monkeypatch.setenv("SECRLENV_DAEMON_URL", "http://127.0.0.1:8765")
    monkeypatch.setenv("SECRLENV_TASK_PACK_SHA256", expected)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return BytesIO(
                json.dumps({"ok": True, "task_pack_sha256": expected}).encode()
            ).read()

    monkeypatch.setattr(
        "yeto_miles_secrlenv.client.urlopen", lambda *_a, **_k: Response()
    )
    require_daemon_ready(timeout_seconds=0.1)

    monkeypatch.setenv("SECRLENV_TASK_PACK_SHA256", "b" * 64)
    with pytest.raises(EpisodeTransportError, match="task-pack identity"):
        require_daemon_ready(timeout_seconds=0.1)


def test_create_retry_default_covers_sixty_rollouts_at_capacity_sixteen(
    monkeypatch,
):
    monkeypatch.delenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", raising=False)
    observed = []

    def positive_env(name, default):
        observed.append((name, default))
        return 0.001

    class UnavailableClient:
        calls = 0

        async def create(self, *_args):
            self.calls += 1
            raise EpisodeTransportError("offline")

    client = UnavailableClient()
    monkeypatch.setattr(agent, "_positive_env", positive_env)
    with pytest.raises(EpisodeTransportError, match="offline"):
        asyncio.run(
            agent._create_with_capacity_retry(
                client, "CVE-2024-1234", "l2"
            )
        )
    assert client.calls == 1
    assert observed == [("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", 14400.0)]
    rollout_count = 60
    max_active = 16
    rollout_timeout_seconds = 3600.0
    admission_waves = (rollout_count + max_active - 1) // max_active
    assert observed[0][1] >= admission_waves * rollout_timeout_seconds


def test_create_retries_capacity_then_one_infrastructure_error_same_identity(
    monkeypatch,
):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    expected = {"episode_id": "a" * 24}
    responses = [
        EpisodeAPIError(503, "capacity_reached", "busy"),
        EpisodeAPIError(503, "infrastructure_error", "provisioning failed"),
        expected,
    ]
    calls = []
    delays = []

    class RecoveringClient:
        async def create(self, task_id, tier):
            calls.append((task_id, tier))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    async def no_wait(delay):
        delays.append(delay)

    monkeypatch.setattr(agent.asyncio, "sleep", no_wait)
    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 0.5)
    result = asyncio.run(
        agent._create_with_capacity_retry(
            RecoveringClient(), "CVE-2024-1234", "l2"
        )
    )
    assert result == expected
    assert calls == [("CVE-2024-1234", "l2")] * 3
    assert delays == [0.5, 0.5]


def test_capacity_retry_stops_at_the_configured_deadline(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "0.02")
    calls = 0

    class BusyClient:
        async def create(self, *_args):
            nonlocal calls
            calls += 1
            raise EpisodeAPIError(503, "capacity_reached", "busy")

    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 0.005)
    with pytest.raises(EpisodeAPIError) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                BusyClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value.code == "capacity_reached"
    assert 2 <= calls < 20


def test_capacity_retry_rejects_success_exactly_at_deadline(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "1")
    clock = [0.0]
    failure = EpisodeAPIError(503, "capacity_reached", "busy")
    calls = []

    class DeadlineClient:
        async def create(self, *_args):
            calls.append(clock[0])
            if len(calls) == 1:
                raise failure
            clock[0] = 1.0
            return {"episode_id": "a" * 24}

    async def advance(delay):
        clock[0] += delay

    monkeypatch.setattr(
        agent, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(agent.asyncio, "sleep", advance)
    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 0.1)
    with pytest.raises(EpisodeAPIError) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                DeadlineClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value is failure
    assert calls == [0.0, 0.1]


def test_capacity_retry_does_not_create_at_the_deadline(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "1")
    clock = [0.0]
    failure = EpisodeAPIError(503, "capacity_reached", "busy")
    calls = 0

    class DeadlineClient:
        async def create(self, *_args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise AssertionError("a retry began at or after its deadline")
            raise failure

    async def advance(delay):
        clock[0] += delay

    monkeypatch.setattr(
        agent, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(agent.asyncio, "sleep", advance)
    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 3.0)
    with pytest.raises(EpisodeAPIError) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                DeadlineClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value is failure
    assert calls == 1


def test_capacity_retry_bounds_a_blocked_retry_by_remaining_time(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "0.05")
    failure = EpisodeAPIError(503, "capacity_reached", "busy")
    calls = 0
    retry_cancelled = False

    class BlockingClient:
        async def create(self, *_args):
            nonlocal calls, retry_cancelled
            calls += 1
            if calls == 1:
                raise failure
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                retry_cancelled = True
                raise

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(agent.asyncio, "sleep", no_wait)
    with pytest.raises(EpisodeAPIError) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                BlockingClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value is failure
    assert calls == 2
    assert retry_cancelled is True


def test_capacity_retry_preserves_external_cancellation(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    retry_started = asyncio.Event()
    retry_cancelled = False
    calls = 0

    class BlockingClient:
        async def create(self, *_args):
            nonlocal calls, retry_cancelled
            calls += 1
            if calls == 1:
                raise EpisodeAPIError(503, "capacity_reached", "busy")
            retry_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                retry_cancelled = True
                raise

    async def no_wait(_delay):
        return None

    async def drive_and_cancel():
        task = asyncio.create_task(
            agent._create_with_capacity_retry(
                BlockingClient(), "CVE-2024-1234", "l2"
            )
        )
        await retry_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(agent.asyncio, "sleep", no_wait)
    asyncio.run(drive_and_cancel())
    assert calls == 2
    assert retry_cancelled is True


def test_infrastructure_error_is_retried_at_most_once(monkeypatch):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    calls = 0
    delays = []

    class BrokenClient:
        async def create(self, *_args):
            nonlocal calls
            calls += 1
            raise EpisodeAPIError(
                503, "infrastructure_error", "provisioning failed"
            )

    async def no_wait(delay):
        delays.append(delay)

    monkeypatch.setattr(agent.asyncio, "sleep", no_wait)
    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 0.5)
    with pytest.raises(EpisodeAPIError) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                BrokenClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value.code == "infrastructure_error"
    assert calls == 2
    assert delays == [0.5]


@pytest.mark.parametrize(
    "failure",
    [
        EpisodeAPIError(400, "invalid_request", "bad task"),
        EpisodeAPIError(400, "capacity_reached", "wrong status"),
        EpisodeAPIError(500, "internal_error", "unexpected failure"),
        EpisodeTransportError("ambiguous transport failure"),
    ],
    ids=("invalid-request", "wrong-capacity-status", "internal-error", "transport"),
)
def test_create_does_not_retry_nonretryable_or_transport_errors(
    monkeypatch, failure
):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    calls = 0

    class FailingClient:
        async def create(self, *_args):
            nonlocal calls
            calls += 1
            raise failure

    async def unexpected_wait(_delay):
        raise AssertionError("single-shot failures must not back off")

    monkeypatch.setattr(agent.asyncio, "sleep", unexpected_wait)
    with pytest.raises(type(failure)) as caught:
        asyncio.run(
            agent._create_with_capacity_retry(
                FailingClient(), "CVE-2024-1234", "l2"
            )
        )
    assert caught.value is failure
    assert calls == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_create_retry_rejects_invalid_capacity_deadline_before_create(
    monkeypatch, value
):
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", value)

    class UnexpectedClient:
        async def create(self, *_args):
            raise AssertionError("invalid retry configuration must fail first")

    with pytest.raises(ValueError, match="SECRLENV_CAPACITY_MAX_WAIT_SECONDS"):
        asyncio.run(
            agent._create_with_capacity_retry(
                UnexpectedClient(), "CVE-2024-1234", "l2"
            )
        )
