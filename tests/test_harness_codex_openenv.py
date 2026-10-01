"""Ports-path acceptance for the Codex Terminal-Bench harness (change rl-codex-harness-rollout).

Covers the six contracts listed in CODEX-PROGRESS §1.3 on top of the moved
legacy suite (tests/test_secrlenv_codex_harness.py):
1. Responses tool call -> TITO append-only prefix reuse on the Miles wire;
2. rollout-level cancellation cleanup (worker process group, lease, tool-wait);
3. reward verification at all three points;
4. mask / token / logprob alignment;
5. sibling segments share one reward and count once;
6. compaction enabled -> preflight fails closed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

import test_secrlenv_codex_harness as legacy_tests
from yeto.rl import tbench_outcome
from yeto.rl.harness.codex import (
    alignment,
    codex_harness_agent as harness,
    codex_openenv_agent_function as adapter,
    codex_openenv_generate as generate_wrapper,
    codex_openenv_subprocess_agent_function as subprocess_agent,
    preflight,
    tbench_reward,
)
from yeto.rl.harness.codex.environment import FakeTerminalEnvironment, serve_environment

KEY = "k" * 48


class _Board:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def enter(self, trajectory_id: str) -> None:
        self.events.append(("enter", trajectory_id))

    def exit(self, trajectory_id: str) -> None:
        self.events.append(("exit", trajectory_id))

    @property
    def open(self) -> int:
        return sum(1 for e, _ in self.events if e == "enter") - sum(1 for e, _ in self.events if e == "exit")


class _Provider:
    def __init__(self, env: FakeTerminalEnvironment, *, deadline: float = 30.0, fail: bool = False) -> None:
        self.env = env
        self.deadline = deadline
        self.fail = fail
        self.destroyed = 0
        self.runner = None

    async def acquire(self, task_id: str, trajectory_id: str) -> subprocess_agent.EnvironmentLease:
        if self.fail:
            raise OSError("sandbox quota exhausted")
        self.runner, url = await serve_environment(self.env, "tok")

        async def destroy() -> None:
            self.destroyed += 1
            await self.runner.cleanup()

        async def describe() -> str:
            return "gone" if self.destroyed else "live"

        return subprocess_agent.EnvironmentLease(
            env_url=url, env_token="tok", verifier=self.env, destroy=destroy, describe=describe,
            deadline_seconds=self.deadline,
        )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- identity / preflight (6)

def test_identity_matches_stock_harness_and_profile():
    identity = adapter.codex_openenv_harness_identity()
    assert set(identity) == {
        "base_instructions_sha256", "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256", "dynamic_tools_schema_sha256",
    }
    assert adapter._OPENENV_IDENTITY_ENV["YETO_CODEX_OPENENV_BACKEND_PROFILE"] == "qwen35_08b"
    assert adapter.stock.BACKEND_MODEL == "qwen35"
    assert callable(adapter.stock._attest_runtime)


def _good_env() -> dict[str, str]:
    return dict(adapter._OPENENV_IDENTITY_ENV)


def test_preflight_fails_closed_on_compaction_scripted_driver_identity_and_version(tmp_path):
    ok = lambda: tmp_path / "codex"  # noqa: E731
    report = preflight.preflight_codex_openenv(_good_env(), attest_runtime=ok, validate_key=lambda: None)
    assert report["compaction"] == "disabled" and report["codex_version"] == "codex-cli 0.145.0"
    for env, match in (
        ({**_good_env(), "YETO_CODEX_COMPACTION_ENABLED": "1"}, "incompatible"),
        ({**_good_env(), "YETO_CODEX_MAX_COMPACTIONS": "3"}, "compaction tuning"),
        ({**_good_env(), "YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER": "1"}, "must not be set"),
        ({k: v for k, v in _good_env().items() if "MODEL_REVISION" not in k}, "identity drifted"),
    ):
        with pytest.raises(preflight.PreflightError, match=match):
            preflight.preflight_codex_openenv(env, attest_runtime=ok, validate_key=lambda: None)

    def bad_version() -> Path:
        raise harness.CodexHarnessError("pinned Codex version contract drifted")

    with pytest.raises(preflight.PreflightError, match="version contract drifted"):
        preflight.preflight_codex_openenv(_good_env(), attest_runtime=bad_version, validate_key=lambda: None)

    def bad_key() -> None:
        raise tbench_outcome.UntrustedTBenchOutcome("key too short")

    with pytest.raises(preflight.PreflightError, match="key source rejected"):
        preflight.preflight_codex_openenv(_good_env(), attest_runtime=ok, validate_key=bad_key)


# ---------------------------------------------------------------- Responses -> TITO prefix (1)

def test_responses_tool_call_round_trip_is_append_only_on_the_miles_wire(monkeypatch):
    legacy_tests._set_bridge_env(monkeypatch)

    async def scenario() -> None:
        runner, miles_url, miles_requests = await legacy_tests._fake_miles([
            legacy_tests._completion("terminal.exec", '{ "command": "git status" }', "call-1", "look", 25),
            legacy_tests._completion("submit", '{"evidence":"done"}', "call-2", "finish", 40),
        ])
        metrics = harness.legacy.AgentMetrics()
        try:
            async with harness._ResponsesBridge(
                miles_url, "fix the repo", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "stream": True, "n": 1},
                metrics, max_seq_len=512,
            ) as bridge:
                headers = {"Authorization": f"Bearer {bridge.token}"}
                initial = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the repo"}]}]
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(initial), headers=headers) as r:
                        assert r.status == 200
                        legacy_tests._sse_events(await r.text())
                    bridge.expect_tool_output("call-1", '{"exit_code":0,"output":"clean","timed_out":false,"truncated":false}')
                    second = copy.deepcopy(bridge._expected_input)
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(second), headers=headers) as r:
                        assert r.status == 200
                    # Append-only: request 2's messages == request 1's + [assistant, tool]
                    m1, m2 = miles_requests[0]["messages"], miles_requests[1]["messages"]
                    assert m2[: len(m1)] == m1
                    appended = m2[len(m1):]
                    assert [m["role"] for m in appended] == ["assistant", "tool"]
                    assert appended[0]["tool_calls"][0]["id"] == "call-1"
                    assert appended[1]["tool_call_id"] == "call-1"
                    # Mutated history is refused (no third Miles sample is taken)
                    mutated = copy.deepcopy(second)
                    mutated[0]["content"][0]["text"] = "fix the repo!"
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(mutated), headers=headers) as r:
                        assert r.status == 400
                    assert len(miles_requests) == 2
        finally:
            await runner.cleanup()

    _run(scenario())


# ---------------------------------------------------------------- subprocess entry: signing, tool-wait, cleanup (2)

def _configure(monkeypatch, env: FakeTerminalEnvironment, **kw) -> tuple[_Provider, _Board]:
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", KEY)
    monkeypatch.setenv("YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER", "1")
    provider = _Provider(env, **kw)
    board = _Board()
    subprocess_agent.configure(provider=provider, tool_wait_board=board)
    return provider, board


def _metadata(**extra) -> dict[str, Any]:
    return {"task_id": "fix-git", "trajectory_id": "traj-1", "prompt": "fix the repo",
            "codex_openenv_driver": "scripted", "script": ["git status", "git checkout -- ."], **extra}


def test_subprocess_run_scrubs_key_relays_tool_wait_and_signs(monkeypatch):
    env = FakeTerminalEnvironment(passed=True)
    provider, board = _configure(monkeypatch, env)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(expected_policy_version="pv-7")))
    assert result is not None and tbench_outcome.MAC_KEY in result
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert reward == 1.0 and outcome["status"] == "completed" and outcome["task_id"] == "fix-git"
    assert env.evaluations == 1 and [c for _, c in env.calls] == ["git status", "git checkout -- ."]
    assert board.events == [("enter", "traj-1"), ("exit", "traj-1")] * 3 and board.open == 0
    assert provider.destroyed == 1
    assert result["chains_total"] == 1 and result["reward_scope"] == "trajectory"
    assert result["expected_policy_version"] == "pv-7"


def test_subprocess_negative_reward_and_timeout_are_signed_policy_boundaries(monkeypatch):
    env = FakeTerminalEnvironment(passed=False)
    _configure(monkeypatch, env)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata()))
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert reward == 0.0 and outcome["status"] == "completed"

    env = FakeTerminalEnvironment(passed=True)
    provider, board = _configure(monkeypatch, env, deadline=0.5)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(hang_seconds=20)))
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert reward == 0.0 and outcome["status"] == "timeout" and outcome["verifier"] == tbench_outcome.TIMEOUT_VERIFIER
    assert env.evaluations == 0 and provider.destroyed == 1 and board.open == 0


def test_subprocess_infrastructure_failures_are_unsigned_and_aborted(monkeypatch):
    env = FakeTerminalEnvironment(passed=True)
    _configure(monkeypatch, env, fail=True)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata()))
    assert tbench_reward.INFRASTRUCTURE_KEY in result and tbench_outcome.MAC_KEY not in result
    sample = SimpleNamespace(metadata=result, status=None)
    assert _run(tbench_reward.reward_func(None, sample)) == 0.0 and sample.status == "ABORTED"

    # worker refuses to start with a reward key visible
    proc = subprocess.run(
        [sys.executable, "-m", subprocess_agent.WORKER_MODULE], input="{}\n", capture_output=True, text=True,
        env={**os.environ, "TBENCH_REWARD_HMAC_KEY": KEY},
    )
    assert proc.returncode == 1 and "leaked" in json.loads(proc.stdout.strip().splitlines()[-1])["reason"]


def test_rollout_cancellation_kills_worker_and_releases_everything(monkeypatch):
    env = FakeTerminalEnvironment(passed=True)
    provider, board = _configure(monkeypatch, env)
    terminated: list[int | None] = []
    original = subprocess_agent._terminate

    async def recording(process):
        await original(process)
        terminated.append(process.returncode)

    monkeypatch.setattr(subprocess_agent, "_terminate", recording)

    async def scenario() -> None:
        task = asyncio.create_task(subprocess_agent.run("http://miles", "p", {}, _metadata(hang_seconds=30)))
        while not board.events:
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())
    assert terminated and terminated[0] is not None  # worker reaped
    assert board.open == 0 and provider.destroyed == 1 and env.evaluations == 0


# ---------------------------------------------------------------- reward verification at three points (3)

def _signed(reward: float = 1.0, **over) -> dict[str, Any]:
    md = tbench_outcome.build_signed_metadata(
        task_id="fix-git", sample_id="s1", episode_id="ep1", status="completed", reward=reward,
        verifier=tbench_outcome.TEST_SH_VERIFIER, testsh_rc=0 if reward else 1, key=KEY,
    )
    md.update(over)
    return md


def test_reward_is_verified_at_reward_function_parent_and_evidence(monkeypatch):
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", KEY)
    good = SimpleNamespace(metadata=_signed(), status=None, rollout_id=1)
    assert _run(tbench_reward.reward_func(None, [good])) == [1.0]
    tampered = copy.deepcopy(good.metadata)
    tampered[tbench_outcome.OUTCOME_KEY]["reward"] = 0.0
    tampered[tbench_outcome.OUTCOME_KEY]["passed"] = False
    for bad in (tampered, {**_signed(), "extra": 1} | {tbench_outcome.OUTCOME_KEY: {**_signed()[tbench_outcome.OUTCOME_KEY], "x": 1}}):
        with pytest.raises(tbench_outcome.UntrustedTBenchOutcome):
            _run(tbench_reward.reward_func(None, SimpleNamespace(metadata=bad, status=None)))
        with pytest.raises(tbench_outcome.UntrustedTBenchOutcome):
            tbench_outcome.verified_outcome(bad)  # parent-side verification point
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", "w" * 48)  # wrong key at the evidence point
    from yeto.rl import trajectory_evidence

    with pytest.raises(Exception):
        trajectory_evidence._verified_outcome(good.metadata, "tbench")


# ---------------------------------------------------------------- siblings share one reward (5)

def test_sibling_segments_share_reward_and_count_once(monkeypatch):
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", KEY)
    a = SimpleNamespace(metadata=_signed(1.0, trajectory_id="t1"), status=None, rollout_id=3)
    b = SimpleNamespace(metadata=_signed(1.0, trajectory_id="t1"), status=None, rollout_id=3)
    c = SimpleNamespace(metadata=_signed(0.0, trajectory_id="t2"), status=None, rollout_id=4)
    rewards = tbench_reward.check_siblings([a, b, c])
    assert rewards == {("rollout", 3): 1.0, ("rollout", 4): 0.0}  # two trajectories, not three samples
    assert tbench_reward.check_group(None, [[a, b], c]).keep is True
    assert tbench_reward.check_group(None, [[a, b]]).keep is False  # zero std over trajectories
    b_bad = SimpleNamespace(metadata=_signed(0.0, trajectory_id="t1"), status=None, rollout_id=3)
    with pytest.raises(tbench_outcome.UntrustedTBenchOutcome, match="different outcomes"):
        tbench_reward.check_group(None, [[a, b_bad], c])
    infra = SimpleNamespace(metadata={tbench_reward.INFRASTRUCTURE_KEY: "x"}, status=None)
    out = tbench_reward.check_group(None, [a, infra])
    assert out.keep is False and out.reason == "group_has_aborted" and infra.status == "ABORTED"


def _sample(tokens, mask, logprobs, response_length, spans=None, **extra):
    wv = [SimpleNamespace(spans=[SimpleNamespace(version=v, abs_start=s, abs_end=e) for v, s, e in spans])] if spans else []
    return SimpleNamespace(tokens=tokens, loss_mask=mask, rollout_log_probs=logprobs, response_length=response_length,
                           weight_versions=wv, metadata={}, status=None, **extra)


def test_generate_wrapper_sets_sibling_keys_checks_policy_version_and_alignment():
    inp = SimpleNamespace(sample=SimpleNamespace(group_index=9, index=42, metadata={"expected_policy_version": "pv"}))
    s1 = _sample([1, 2, 3, 4], [0, 0, 1, 1], [-0.1, -0.2], 2, [("pv", 2, 4)])
    s2 = _sample([1, 5, 6], [0, 1, 1], [-0.3, -0.4], 2, [("pv", 1, 3)])

    async def upstream(_input):
        return SimpleNamespace(samples=[s1, s2])

    out = _run(generate_wrapper.generate(inp, upstream=upstream))
    assert [s.group_index for s in out.samples] == [9, 9] and [s.rollout_id for s in out.samples] == [42, 42]
    assert [s.metadata["chain_index"] for s in out.samples] == [0, 1] and s1.metadata["chains_total"] == 2
    assert s1.status is None and s1.metadata["policy_versions_actual"] == ["pv"]

    stale = _sample([1, 2, 3], [0, 1, 1], [-0.1, -0.2], 2, [("pv-old", 1, 3)])
    _run(generate_wrapper.generate(inp, upstream=lambda _i: _coro(SimpleNamespace(samples=[stale]))))
    assert stale.status == "ABORTED" and stale.metadata["policy_age_violation"] == 1
    assert tbench_outcome.MAC_KEY not in stale.metadata and "policy_age" in stale.metadata[tbench_reward.INFRASTRUCTURE_KEY]


async def _coro(value):
    return value


# ---------------------------------------------------------------- mask / token / logprob alignment (4)

def test_alignment_contract_fails_closed():
    good = _sample([7, 8, 9, 10, 11], [0, 0, 1, 0, 1], [-0.5, 0.0, -0.1], 3, [("v", 2, 3), ("v", 4, 5)])
    alignment.assert_sample_alignment(good)
    cases = {
        "neither": _sample([7, 8, 9], [1, 1], [-0.1, -0.1, -0.1], 3),
        "rollout_log_probs": _sample([7, 8, 9], [0, 1, 1], None, 2),
        "prompt position": _sample([7, 8, 9], [1, 0, 1], [-0.1, -0.1], 2),
        "finite rollout logprob": _sample([7, 8, 9], [0, 1, 1], [None, -0.1], 2),
        "outside generated": _sample([7, 8, 9, 10], [0, 1, 1, 1], [-0.1, -0.1, -0.1], 3, [("v", 1, 3)]),
    }
    for match, sample in cases.items():
        with pytest.raises(alignment.AlignmentError, match=match):
            alignment.assert_sample_alignment(sample)


# ---------------------------------------------------------------- 2.2 legacy forwarding / 6.3 boundary

def test_legacy_preflight_forwarder_matches_legacy_failure_classes(monkeypatch):
    env = dict(adapter._OPENENV_IDENTITY_ENV)
    preflight.forward_legacy_openenv_preflight(None, "qwen35_08b", env)
    with pytest.raises(ValueError, match="requires backend profile"):
        preflight.forward_legacy_openenv_preflight(None, "qwen35", env)
    with pytest.raises(ValueError, match="environment drifted"):
        preflight.forward_legacy_openenv_preflight(None, "qwen35_08b", {k: v for k, v in env.items() if "MODEL_REVISION" not in k})
    pins = preflight.required_pin_updates()
    assert pins["CODEX_HARNESS_AGENT_SHA256"] == "995e48f0e2817191314f19e794e25fd70e738aec5957213b51914f24d552f7b7"
    assert pins["CODEX_OPENENV_AGENT"].endswith("codex_openenv_subprocess_agent_function.run")


def test_scrubbed_environment_removes_every_reward_key_name():
    base = {n: "x" for n in adapter.hmac_key_env_names()} | {"PATH": "/bin"}
    assert subprocess_agent.scrubbed_environment(base) == {"PATH": "/bin"}
    with pytest.raises(RuntimeError, match="leaked"):
        adapter.assert_no_reward_key(base)
