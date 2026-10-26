"""S25 N10.1-N10.4: neutral agentic layer, Harbor harness, Miles and verl adapters (CPU, fakes only).

No Harbor, Miles, verl or sandbox is needed: Harbor's ``Trial`` is a fake class,
Miles' example module and verl's agent-loop module are stubbed in ``sys.modules``.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.agentic import (
    OUTCOME_KEY,
    AdmissionGate,
    AgentBudget,
    AgentTask,
    HarnessConfigError,
    Outcome,
    TokenAlignmentError,
    TokenTurn,
    TrialVerdict,
    TrustViolation,
    classify,
    join_turns,
    run_agent_task,
)
from yeto.rl.agentic.trust import check_agent_env, task_dir


def _task(**over) -> AgentTask:
    return AgentTask(**{"task_id": "fix-git", "agent": "codex", "provider": "modal", **over})


def _turn(prompt, completion, lp=None) -> TokenTurn:
    return TokenTurn(tuple(prompt), tuple(completion), tuple(lp or [-0.5] * len(completion)))


# --- neutral types and token join (N10.2 / N10.3) ----------------------------


def test_join_turns_masks_environment_tokens_and_keeps_policy_logprobs():
    turns = [_turn([1, 2, 3], [10, 11], [-0.1, -0.2]),
             _turn([1, 2, 3, 10, 11, 20, 21, 22], [12], [-0.3])]
    seg = join_turns(turns)
    assert seg.prompt_ids == (1, 2, 3)
    assert seg.response_ids == (10, 11, 20, 21, 22, 12)
    assert seg.response_mask == (1, 1, 0, 0, 0, 1)
    assert seg.response_logprobs == (-0.1, -0.2, 0.0, 0.0, 0.0, -0.3)
    # every mask-1 token is a generated token, in order, with its own logprob
    generated = [t for t, m in zip(seg.response_ids, seg.response_mask) if m]
    assert generated == [10, 11, 12]
    assert not seg.truncated


def test_join_turns_refuses_rewritten_history_and_empty_sessions():
    with pytest.raises(TokenAlignmentError, match="no model calls"):
        join_turns([])
    with pytest.raises(TokenAlignmentError, match="history rewritten at token 3"):
        join_turns([_turn([1, 2, 3], [10]), _turn([1, 2, 3, 99, 5], [6])])
    with pytest.raises(TokenAlignmentError, match="logprobs"):
        TokenTurn((1,), (2, 3), (-0.1,))
    with pytest.raises(TokenAlignmentError, match="finite"):
        TokenTurn((1,), (2,), (float("nan"),))


def test_join_turns_truncates_to_the_trainer_response_width():
    seg = join_turns([_turn([1], [2, 3]), _turn([1, 2, 3, 4], [5, 6])], max_response_len=4)
    assert seg.response_ids == (2, 3, 4, 5) and seg.response_mask == (1, 1, 0, 1)
    assert seg.truncated


def test_agent_task_from_miles_metadata_and_safe_ids():
    task = AgentTask.from_mapping({"instance_id": "fix-git", "agent_name": "codex", "max_seq_len": 4096},
                                  defaults={"provider": "modal", "wall_clock_s": 600})
    assert (task.task_id, task.agent, task.provider) == ("fix-git", "codex", "modal")
    assert task.budget == AgentBudget(wall_clock_s=600.0, max_seq_len=4096)
    with pytest.raises(ValueError, match="safe identifier"):
        _task(task_id="../etc")
    with pytest.raises(ValueError, match="below wall_clock_s"):
        AgentBudget(agent_timeout_s=900, wall_clock_s=600)


# --- outcome classification and trust boundary (N10.2) ------------------------


@pytest.mark.parametrize(("verdict", "expected"), [
    (TrialVerdict(rewards={"reward": 1.0}), (Outcome.OK, 1.0, "verified")),
    (TrialVerdict(rewards={"score": 0}), (Outcome.OK, 0.0, "verified")),
    (TrialVerdict(rewards=None, exception_type="AgentTimeoutError"),
     (Outcome.POLICY_LIMIT, 0.0, "AgentTimeoutError")),
    (TrialVerdict(rewards={"reward": 1.0}, exception_type="MaxSeqLenExceededError"),
     (Outcome.POLICY_LIMIT, 1.0, "MaxSeqLenExceededError")),
    (TrialVerdict(rewards={"reward": 1.0}, exception_type="EnvironmentStartTimeoutError"),
     (Outcome.ABORTED, 0.0, "EnvironmentStartTimeoutError")),
    (TrialVerdict(rewards=None, exception_type="SomeFutureHarborError"),
     (Outcome.ABORTED, 0.0, "SomeFutureHarborError")),
    (TrialVerdict(rewards=None), (Outcome.ABORTED, 0.0, "no_verifier_result")),
    (TrialVerdict(rewards={}), (Outcome.ABORTED, 0.0, "no_verifier_result")),
])
def test_classify_never_turns_infrastructure_into_a_zero_reward(verdict, expected):
    assert classify(verdict) == expected


@pytest.mark.parametrize("rewards", [{"reward": 2.0}, {"reward": float("inf")}, {"reward": True},
                                     {"a": 1.0, "b": 0.0}])
def test_classify_refuses_untrusted_verifier_rewards(rewards):
    with pytest.raises(TrustViolation):
        classify(TrialVerdict(rewards=rewards))


def test_trust_boundary_env_and_task_dir(tmp_path):
    assert check_agent_env({"OPENAI_BASE_URL": "http://x", "OPENAI_API_KEY": "dummy"})
    with pytest.raises(TrustViolation, match="WANDB_API_KEY"):
        check_agent_env({"OPENAI_BASE_URL": "http://x", "WANDB_API_KEY": "k"})
    (tmp_path / "fix-git").mkdir()
    assert task_dir(tmp_path, _task()) == (tmp_path / "fix-git").resolve()
    with pytest.raises(FileNotFoundError):
        task_dir(tmp_path, _task(task_id="missing"))
    (tmp_path / "out").symlink_to("/")
    with pytest.raises(TrustViolation, match="escapes"):
        task_dir(tmp_path, _task(task_id="out"))


# --- outer layer runner (N10.2) ----------------------------------------------


class _Harness:
    def __init__(self, verdict=None, exc=None, delay=0.0):
        self.verdict, self.exc, self.delay, self.calls = verdict, exc, delay, []

    async def run(self, task, endpoint):
        self.calls.append((task.task_id, endpoint))
        await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.verdict


def _run(coro):
    return asyncio.run(coro)


def test_runner_classifies_and_counts():
    gate = AdmissionGate(2, policy_version=lambda: 7)
    traj = _run(run_agent_task(_task(), _Harness(TrialVerdict(rewards={"reward": 1})), "http://u/v1", gate))
    assert (traj.outcome, traj.reward, traj.policy_version) == (Outcome.OK, 1.0, 7)
    assert traj.to_metadata()[OUTCOME_KEY] == "ok"
    snap = gate.metrics.snapshot()
    assert snap["harness_outcome_ok_total"] == 1 and snap["harness_in_flight"] == 0 and snap["env_live"] == 0


def test_runner_aborts_on_harness_failure_wall_clock_and_drain():
    gate = AdmissionGate(1)
    traj = _run(run_agent_task(_task(), _Harness(exc=ConnectionError("sandbox lost")), "u", gate))
    assert (traj.outcome, traj.reason) == (Outcome.ABORTED, "ConnectionError")
    slow = _task(budget=AgentBudget(wall_clock_s=0.05))
    traj = _run(run_agent_task(slow, _Harness(TrialVerdict(rewards={"reward": 1}), delay=1.0), "u", gate))
    assert (traj.outcome, traj.reason) == (Outcome.ABORTED, "wall_clock_cap")
    gate.drain()
    harness = _Harness(TrialVerdict(rewards={"reward": 1}))
    traj = _run(run_agent_task(_task(), harness, "u", gate))
    assert (traj.outcome, traj.reason, harness.calls) == (Outcome.ABORTED, "admission_draining", [])
    assert gate.metrics.snapshot()["harness_outcome_aborted_total"] == 3


def test_runner_raises_configuration_and_trust_errors():
    gate = AdmissionGate(1)
    with pytest.raises(HarnessConfigError):
        _run(run_agent_task(_task(), _Harness(exc=HarnessConfigError("no task dir")), "u", gate))
    with pytest.raises(TrustViolation):
        _run(run_agent_task(_task(), _Harness(TrialVerdict(rewards={"reward": 5})), "u", gate))


def test_gate_bounds_sandboxes_in_flight():
    gate = AdmissionGate(2)
    peak = []

    class Watch(_Harness):
        async def run(self, task, endpoint):
            peak.append(gate.metrics.in_flight)
            return await super().run(task, endpoint)

    async def many():
        h = Watch(TrialVerdict(rewards={"reward": 0}), delay=0.02)
        await asyncio.gather(*(run_agent_task(_task(), h, "u", gate) for _ in range(6)))

    _run(many())
    assert max(peak) == 2 and gate.metrics.admitted == 6


# --- Harbor harness (N10.1) --------------------------------------------------


def _result(rewards=None, exc_type=None):
    t0 = datetime(2026, 10, 19)
    phase = SimpleNamespace(started_at=t0, finished_at=t0 + timedelta(seconds=3))
    return SimpleNamespace(
        verifier_result=None if rewards is None else SimpleNamespace(rewards=rewards),
        exception_info=None if exc_type is None else SimpleNamespace(exception_type=exc_type,
                                                                    exception_message="m"),
        started_at=t0, finished_at=t0 + timedelta(seconds=10),
        environment_setup=phase, agent_setup=None, agent_execution=phase, verifier=None,
        agent_result=SimpleNamespace(n_input_tokens=100, n_output_tokens=20, n_steps=4),
    )


class _FakeTrial:
    """Shape of harbor.trial.trial.Trial: ``await Trial.create(config)``, ``await trial.run()``."""

    result = None
    configs: list = []

    def __init__(self, config):
        self.config = config
        self.paths = SimpleNamespace(trial_dir=Path("/tmp/harbor_trials/t1"))

    @classmethod
    async def create(cls, config):
        cls.configs.append(config)
        return cls(config)

    async def run(self):
        return type(self).result


def test_harbor_harness_reads_the_trial_result():
    from yeto.rl.harness.harbor import HarborHarness

    _FakeTrial.result, _FakeTrial.configs = _result({"reward": 1.0}), []
    harness = HarborHarness(lambda task, url: {"task": task.task_id, "url": url}, trial_cls=_FakeTrial)
    verdict = _run(harness.run(_task(), "http://rec/v1"))
    assert _FakeTrial.configs == [{"task": "fix-git", "url": "http://rec/v1"}]
    assert verdict.rewards == {"reward": 1.0} and verdict.exception_type is None
    assert verdict.timings_s == {"total_time": 10.0, "env_setup_time": 3.0, "agent_run_time": 3.0}
    assert verdict.agent_metrics == {"n_input_tokens": 100, "n_output_tokens": 20, "n_steps": 4}
    assert verdict.trial_dir == "/tmp/harbor_trials/t1"


def test_harbor_pins_and_agent_env():
    from yeto.rl.harness.harbor import HARBOR_COMMIT, agent_env, install_spec

    assert len(HARBOR_COMMIT) == 40 and install_spec().endswith("@" + HARBOR_COMMIT)
    assert agent_env("codex", "http://r/v1")["OPENAI_BASE_URL"] == "http://r/v1"
    with pytest.raises(HarnessConfigError, match="claude-code"):
        agent_env("claude-code", "http://r/v1")


# --- Miles adapter (N10.1 / N10.2) -------------------------------------------


@pytest.fixture
def miles_stubs(monkeypatch):
    session = types.ModuleType("miles.rollout.agentic.session")
    session.openai_session_url = lambda base: f"{base}/v1"
    for name in ("miles", "miles.rollout", "miles.rollout.agentic"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.rollout.agentic.session", session)
    example = types.ModuleType("harbor_agent_function")
    example.calls = []
    example._ensure_provider_key = lambda: example.calls.append("key")

    def build_trial_config(metadata, session_url, request_kwargs):
        example.calls.append((metadata["agent_name"], session_url))
        return SimpleNamespace(agent=SimpleNamespace(env={"OPENAI_BASE_URL": session_url}, kwargs={}))

    example.build_trial_config = build_trial_config
    monkeypatch.setitem(sys.modules, "harbor_agent_function", example)
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    mod = importlib.import_module("yeto.rl.adapters.miles.harbor_agent")
    mod._gates.clear()
    return mod, example


def test_miles_agent_function_wraps_the_example_with_the_outer_layer(miles_stubs, monkeypatch):
    mod, example = miles_stubs
    seen = {}

    class Trial(_FakeTrial):
        @classmethod
        async def create(cls, config):
            seen["version"] = config.agent.kwargs["version"]
            seen["summary"] = config.agent.kwargs["reasoning_summary"]
            return cls(config)

    Trial.result = _result({"reward": 1.0})
    monkeypatch.setattr(mod, "HarborHarness",
                        lambda build: __import__("yeto.rl.harness.harbor", fromlist=["x"]).HarborHarness(
                            build, trial_cls=Trial))
    out = _run(mod.run("http://ss:30000/sessions/s1", metadata={"instance_id": "fix-git"}))
    assert example.calls == ["key", ("codex", "http://ss:30000/sessions/s1/v1")]
    assert seen["version"] == "0.145.0"  # Harbor's default would be @latest
    assert seen["summary"] == "none"  # the gateway refuses reasoning summaries
    assert (out[OUTCOME_KEY], out["reward"]) == ("ok", 1.0)

    Trial.result = _result(None, "EnvironmentStartTimeoutError")
    out = _run(mod.run("http://ss", metadata={"instance_id": "fix-git"}))
    assert (out[OUTCOME_KEY], out["reward"]) == ("aborted", 0.0)  # Miles' example would score 0


def test_miles_reward_hook_marks_aborted_and_fails_closed(miles_stubs):
    mod, _ = miles_stubs
    ok = SimpleNamespace(metadata={OUTCOME_KEY: "ok", "reward": 1.0}, status="COMPLETED")
    aborted = SimpleNamespace(metadata={OUTCOME_KEY: "aborted", "reward": 0.0}, status="COMPLETED")
    assert _run(mod.reward_func(None, [ok, aborted])) == [1.0, 0.0]
    assert aborted.status == "ABORTED" and ok.status == "COMPLETED"
    with pytest.raises(mod.UntrustedAgentOutcome):
        _run(mod.reward_func(None, SimpleNamespace(metadata={"reward": 1.0})))


def test_miles_train_args_match_the_example_except_the_yeto_hooks(monkeypatch):
    path = Path("/home/michael/work/miles-next/examples/experimental/harbor/launch_common.py")
    if not path.exists():
        pytest.skip("Miles source not on this machine")
    creds = types.ModuleType("miles.rollout.agentic.credentials")
    creds.PROVIDER_CREDENTIALS, creds.provision_provider = {}, None
    for name in ("miles", "miles.rollout", "miles.rollout.agentic"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.rollout.agentic.credentials", creds)
    spec = importlib.util.spec_from_file_location("_miles_harbor_launch_common", path)
    launch_common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_common)
    from yeto.rl.adapters.miles import harbor_agent

    kw = dict(tito_model="glm5", session_server_workers=4, session_server_external_host="10.0.0.1")
    theirs = launch_common.agentic_train_args(**kw).split()
    ours = harbor_agent.train_args(**kw).split()
    swap = {"harbor_agent_function.run": harbor_agent.AGENT_FUNCTION,
            "generate.reward_func": harbor_agent.REWARD_FUNCTION}
    assert ours == [swap.get(token, token) for token in theirs]


# --- verl adapter (N10.3) ----------------------------------------------------


@pytest.fixture
def verl_loop(monkeypatch):
    """Stub of verl.experimental.agent_loop.agent_loop with the same names and shapes."""
    from pydantic import BaseModel

    mod = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    registry = {}

    class AgentLoopMetrics(BaseModel):
        generate_sequences: float = 0.0
        tool_calls: float = 0.0

    class AgentLoopOutput(BaseModel):
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: list[float] | None = None
        reward_score: float | None = None
        num_turns: int = 0
        metrics: AgentLoopMetrics
        extra_fields: dict = {}

    class AgentLoopBase:
        def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kw):
            self.config = trainer_config
            self.rollout_config = trainer_config.actor_rollout_ref.rollout
            self.tokenizer = tokenizer

    def register(name):
        def deco(cls):
            registry[name] = {"_target_": f"{cls.__module__}.{cls.__qualname__}"}
            return cls
        return deco

    mod.AgentLoopBase, mod.AgentLoopMetrics, mod.AgentLoopOutput, mod.register = (
        AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register)
    mod.registry = registry
    for name in ("verl", "verl.experimental", "verl.experimental.agent_loop"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.agent_loop", mod)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.verl.harbor_agent_loop", raising=False)
    loop_mod = importlib.import_module("yeto.rl.adapters.verl.harbor_agent_loop")
    loop_mod._gates.clear()
    yield loop_mod, mod
    sys.modules.pop("yeto.rl.adapters.verl.harbor_agent_loop", None)  # bound to the stub


class _Recorder:
    def __init__(self, turns):
        self.turns, self.opened, self.closed = turns, [], []

    async def open(self, trajectory_id, sampling_params):
        self.opened.append((trajectory_id, dict(sampling_params)))
        return f"http://recorder/{trajectory_id}/v1"

    async def close(self, trajectory_id):
        self.closed.append(trajectory_id)
        return self.turns


def _make_loop(loop_mod, recorder, harness, *, prompt_length=16, response_length=32):
    cls = type("Loop", (loop_mod.HarborAgentLoop,), {
        "recorder_factory": staticmethod(lambda: recorder),
        "harness_factory": staticmethod(lambda: harness),
    })
    rollout = SimpleNamespace(prompt_length=prompt_length, response_length=response_length)
    cfg = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    return cls(cfg, None, tok, None, None, None)


_ROW = {"extra_info": {"agent_task": {"task_id": "fix-git", "agent": "codex", "provider": "modal",
                                      "trajectory_id": "traj-1"}}}


def test_verl_loop_registers_without_patching_verl(verl_loop):
    loop_mod, stub = verl_loop
    assert stub.registry["yeto_harbor"] == {"_target_": "yeto.rl.adapters.verl.harbor_agent_loop.HarborAgentLoop"}
    assert issubclass(loop_mod.HarborAgentLoop, stub.AgentLoopBase)


def test_verl_loop_token_mask_alignment_with_fake_trial(verl_loop):
    loop_mod, _ = verl_loop
    from yeto.rl.harness.harbor import HarborHarness

    turns = [replace(_turn([5, 6, 7], [40, 41, 42], [-0.1, -0.2, -0.3]), finish_reason="length"),
             _turn([5, 6, 7, 40, 41, 42, 90, 91], [43, 2], [-0.4, -0.5])]
    recorder = _Recorder(turns)
    _FakeTrial.result, _FakeTrial.configs = _result({"reward": 1.0}), []
    harness = HarborHarness(lambda task, url: {"url": url}, trial_cls=_FakeTrial)
    loop = _make_loop(loop_mod, recorder, harness)
    out = _run(loop.run({"temperature": 1.0, "top_p": 1.0}, **_ROW))

    ((session, sampling),) = recorder.opened
    assert session.startswith("traj-1-") and sampling == {"temperature": 1.0, "top_p": 1.0}
    assert recorder.closed == [session]  # one session per sample: verl repeats a row n times
    assert _FakeTrial.configs == [{"url": f"http://recorder/{session}/v1"}]  # agent talks to the recorder
    assert out.prompt_ids == [5, 6, 7]
    assert out.response_ids == [40, 41, 42, 90, 91, 43, 2]
    assert out.response_mask == [1, 1, 1, 0, 0, 1, 1]
    assert out.response_logprobs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5]
    assert len(out.response_ids) == len(out.response_mask) == len(out.response_logprobs)
    # prompt + response is exactly the last call's full stream: trained tokens == generated tokens
    assert out.prompt_ids + out.response_ids == list(turns[-1].prompt_ids + turns[-1].completion_ids)
    assert out.reward_score == 1.0 and out.num_turns == 4
    assert out.extra_fields[OUTCOME_KEY] == "ok" and out.extra_fields["truncated"] is False
    assert out.extra_fields["length_capped_turns"] == 1  # the first call hit its token cap


def test_verl_loop_aborts_on_misaligned_tokens_and_infra_errors(verl_loop):
    loop_mod, _ = verl_loop
    from yeto.rl.harness.harbor import HarborHarness

    bad = [_turn([5, 6], [40]), _turn([5, 6, 77, 1], [41])]  # history rewritten (compaction)
    _FakeTrial.result = _result({"reward": 1.0})
    loop = _make_loop(loop_mod, _Recorder(bad), HarborHarness(lambda t, u: {}, trial_cls=_FakeTrial))

    async def run_and_snapshot():
        return await loop.run({}, **_ROW), loop_mod._gate().metrics.snapshot()  # the gate is per loop

    out, snap = _run(run_and_snapshot())
    assert out.extra_fields[OUTCOME_KEY] == "aborted" and out.extra_fields["outcome_reason"].startswith(
        "token_alignment")
    assert (out.response_mask, out.reward_score) == ([0], 0.0)  # placeholder: no loss tokens
    assert (snap["harness_outcome_ok_total"], snap["harness_outcome_aborted_total"]) == (0, 1)

    _FakeTrial.result = _result(None, "VerifierTimeoutError")
    loop = _make_loop(loop_mod, _Recorder([_turn([5], [6])]), HarborHarness(lambda t, u: {}, trial_cls=_FakeTrial))
    out = _run(loop.run({}, **_ROW))
    assert (out.extra_fields[OUTCOME_KEY], out.response_mask, out.reward_score) == ("aborted", [0], 0.0)

    loop = _make_loop(loop_mod, _Recorder([_turn(list(range(20)), [6])]),
                      HarborHarness(lambda t, u: {}, trial_cls=_FakeTrial), prompt_length=16)
    _FakeTrial.result = _result({"reward": 1.0})
    out = _run(loop.run({}, **_ROW))
    assert out.extra_fields["outcome_reason"] == "prompt_exceeds_prompt_length"


def test_verl_loop_requires_an_agent_task_row(verl_loop):
    loop_mod, _ = verl_loop
    loop = _make_loop(loop_mod, _Recorder([]), _Harness())
    with pytest.raises(HarnessConfigError, match="agent_task"):
        _run(loop.run({}, extra_info={}))


# --- verl refuses Miles agentic flags (N10.4) --------------------------------


def test_verl_rules_refuse_agentic_flags():
    from yeto.rl.adapters.verl import run_config_rules as rules

    rules.check_agentic(SimpleNamespace(use_session_server=False, tito_model=None))
    with pytest.raises(ValueError, match="--custom-agent-function-path, --use-session-server"):
        rules.check_agentic(SimpleNamespace(custom_agent_function_path="a.b.run", use_session_server=True))
    rules.refuse_agentic_argv(["--some-other-flag", "1"])
    with pytest.raises(ValueError, match="--tito-model"):
        rules.refuse_agentic_argv(["--tito-model=glm5"])


def test_verl_launcher_refuses_agentic_flags_before_launch():
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args(["--use-session-server", "--tito-model", "qwen38small"])
    args.rl_backend = "verl"
    with pytest.raises(ValueError, match="verl 后端不支持 Miles agentic 参数"):
        _prepare_rl_args(args)


def test_verl_island_entry_refuses_agentic_argv():
    from yeto.rl.adapters.verl import island_entry

    argv = ["--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0",
            "--reward-function", "f", "--global-rounds", "1", "--groups-per-round", "1",
            "--samples-per-group", "1", "--rollout-max-response-len", "8", "--event-tape", "t",
            "--inner-lr", "1e-5", "--seq-len", "16", "--seed", "1",
            "--custom-agent-function-path", "x.y.run"]
    with pytest.raises(ValueError, match="--custom-agent-function-path"):
        island_entry.main(argv)


def test_miles_custom_provider_uses_the_yeto_builder(miles_stubs, monkeypatch, tmp_path):
    """boat: the example rejects it (EnvironmentType), so yeto builds the TrialConfig."""
    mod, example = miles_stubs
    (tmp_path / "fix-git").mkdir()
    monkeypatch.setenv("HARBOR_ENV_TYPE", "boat")
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"ttl_s": 600}')
    monkeypatch.setenv("AGENT_TIMEOUT", "900")
    built = []

    def yeto_builder(task, endpoint, settings):
        built.append((task.agent, task.provider, task.budget.agent_timeout_s, endpoint,
                      settings.tasks_dir, dict(settings.provider_kwargs)))
        return SimpleNamespace(agent=SimpleNamespace(env={}, kwargs={}))

    monkeypatch.setattr(mod, "build_trial_config", yeto_builder)

    class Trial(_FakeTrial):
        result = _result({"reward": 1.0})

    monkeypatch.setattr(mod, "HarborHarness",
                        lambda build: __import__("yeto.rl.harness.harbor", fromlist=["x"]).HarborHarness(
                            build, trial_cls=Trial))
    out = _run(mod.run("http://ss", metadata={"instance_id": "fix-git", "agent_name": "oracle"}))
    assert built == [("oracle", "boat", 900.0, "http://ss/v1", tmp_path, {"ttl_s": 600})]
    assert example.calls == ["key"]  # the example's builder is not called
    assert (out[OUTCOME_KEY], out["reward"]) == ("ok", 1.0)


def test_miles_builtin_provider_keeps_the_example_builder(miles_stubs):
    mod, example = miles_stubs
    task = mod.task_from_metadata({"instance_id": "fix-git", "agent_name": "terminus-2"})
    assert task.agent == "terminus-2" and task.provider == "modal"
    build = mod.trial_config_builder(task, example, {"instance_id": "fix-git"}, {})
    build(task, "http://ss/v1")
    assert example.calls == [("terminus-2", "http://ss/v1")]


def test_harbor_settings_from_env(monkeypatch, tmp_path):
    from yeto.rl.harness.harbor import HarborSettings

    monkeypatch.delenv("HARBOR_TASKS_DIR", raising=False)
    with pytest.raises(HarnessConfigError, match="HARBOR_TASKS_DIR"):
        HarborSettings.from_env()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_KWARGS", "[1]")
    with pytest.raises(HarnessConfigError, match="JSON object"):
        HarborSettings.from_env()
    monkeypatch.setenv("HARBOR_ENV_KWARGS", "")
    monkeypatch.setenv("HARBOR_TIMEOUT_MULTIPLIER", "2")
    settings = HarborSettings.from_env()
    assert (settings.tasks_dir, settings.provider_kwargs, settings.timeout_multiplier) == (tmp_path, {}, 2.0)
