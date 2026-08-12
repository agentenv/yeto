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


def test_capacity_retry_default_is_bounded_to_two_minutes(monkeypatch):
    monkeypatch.delenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", raising=False)
    observed = []

    def positive_env(name, default):
        observed.append((name, default))
        return 0.001

    class UnavailableClient:
        async def create(self, *_args):
            raise EpisodeTransportError("offline")

    monkeypatch.setattr(agent, "_positive_env", positive_env)
    monkeypatch.setattr(agent.random, "uniform", lambda *_args: 0.0)
    with pytest.raises(EpisodeTransportError, match="offline"):
        asyncio.run(
            agent._create_with_capacity_retry(
                UnavailableClient(), "CVE-2024-1234", "l2"
            )
        )
    assert observed == [("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", 120.0)]
