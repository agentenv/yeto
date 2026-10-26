"""S25 N10.5 on verl: HarborAgentLoop with the records proxy as its token recorder (CPU).

Real pieces: ``HarborAgentLoop.run``, the outer layer, ``ProxyTokenRecorder``
serving a ``RecordsProxy`` over HTTP on 127.0.0.1, the gateway and the
recording backend.  Fakes: verl's agent-loop module (stub, as in
``test_agentic_harbor``), the engine (scripted ``FakeEngine``) and the agent
(a harness that calls the recorder URL over OpenAI Responses, as Harbor's
codex agent would).  Tokenizer: Qwen3-0.6B from the local HF cache (skip if absent).
"""

from __future__ import annotations

import asyncio
import copy
import sys
from typing import Any

import pytest
import test_agentic_harbor as agentic
from test_agentic_harbor import _make_loop
from test_records_proxy import (
    SCRIPT,
    SIGNED,
    ScriptedModel,
    _template,
    echo,
    engine_for,
    responses_body,
    tool_output,
    user_item,
)

from yeto.rl.agentic import OUTCOME_KEY, HarnessConfigError, TrialVerdict
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder, recorder_from_env, turns_from_records

pytest.importorskip("aiohttp")

verl_loop = agentic.verl_loop  # the verl agent-loop stub fixture

ROW = {"extra_info": {"agent_task": {"task_id": "fix-git", "agent": "codex", "provider": "boat",
                                     "trajectory_id": "traj-1"}}}
VERL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}


@pytest.fixture(scope="module")
def qwen():
    return _template("qwen3")


class CodexLikeAgent:
    """Harness stand-in: a Codex-like Responses loop against the recorder URL (as Harbor's codex agent)."""

    def __init__(self, turns: int = 3) -> None:
        self.turns = turns
        self.endpoints: list[str] = []

    async def run(self, task: Any, endpoint: str) -> TrialVerdict:
        import aiohttp

        self.endpoints.append(endpoint)
        items = [user_item("read a.txt")]
        async with aiohttp.ClientSession() as http:
            for turn in range(self.turns):
                async with http.post(f"{endpoint}/responses", json=responses_body(copy.deepcopy(items))) as reply:
                    assert reply.status == 200, await reply.text()
                    response = await reply.json()
                items += echo(response["output"])
                items += [tool_output(o["call_id"], f"result {turn}")
                          for o in response["output"] if o["type"] == "function_call"]
        return TrialVerdict(rewards={"reward": 1.0})


def _proxy(template, script=SCRIPT) -> RecordsProxy:
    return RecordsProxy(engine_for(ScriptedModel(template, copy.deepcopy(script))), template,
                        model="policy", signed_sampling=SIGNED, max_chains=1)


def test_recorder_feeds_the_verl_loop_the_tokens_the_engine_sampled(verl_loop, qwen):
    loop_mod, _ = verl_loop
    proxy = _proxy(qwen)
    recorder = ProxyTokenRecorder(proxy)
    agent = CodexLikeAgent()
    loop = _make_loop(loop_mod, recorder, agent, prompt_length=4096, response_length=4096)

    async def go():
        try:
            out = await loop.run(VERL_SAMPLING, **ROW)
        finally:
            await recorder.aclose()
        return out

    out = asyncio.run(go())
    assert out.extra_fields[OUTCOME_KEY] == "ok" and out.reward_score == 1.0 and out.num_turns == 6
    (endpoint,) = agent.endpoints
    assert endpoint.startswith("http://127.0.0.1:") and "/t/traj-1-" in endpoint
    # prompt + response is the engine's last prompt + its completion: trained == sampled
    engine = proxy.backend.engine
    last_prompt = list(engine.token_requests[-1][0][0])
    assert out.prompt_ids + out.response_ids[:len(last_prompt) - len(out.prompt_ids)] == last_prompt
    assert len(out.response_ids) == len(out.response_mask) == len(out.response_logprobs)
    assert sum(out.response_mask) > 0 and out.response_mask[-1] == 1
    assert all(lp == 0.0 for lp, m in zip(out.response_logprobs, out.response_mask) if m == 0)
    assert out.extra_fields["provenance"]["engine"] == "fake"
    assert proxy.backend.sessions == {} and proxy.gateway.registries == {}  # forgotten after close


def test_one_recorder_per_event_loop_from_the_env_factory(verl_loop, qwen, monkeypatch):
    """verl builds one agent loop per sample; the factory runs once per event loop."""
    loop_mod, _ = verl_loop
    built: list[ProxyTokenRecorder] = []
    module = type(sys)("t_n2w_recorder_factory")

    def make() -> ProxyTokenRecorder:
        built.append(recorder_from_env())
        return built[-1]

    module.make, module.proxy = make, lambda: _proxy(qwen, SCRIPT * 2)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(loop_mod.RECORDER_ENV, f"{module.__name__}:make")
    monkeypatch.setenv("YETO_RECORDS_PROXY", f"{module.__name__}:proxy")
    loop_mod._recorders.clear()
    agent = CodexLikeAgent()
    cls = type("Loop", (loop_mod.HarborAgentLoop,), {"harness_factory": staticmethod(lambda: agent)})
    rollout = agentic.SimpleNamespace(prompt_length=4096, response_length=4096)
    cfg = agentic.SimpleNamespace(actor_rollout_ref=agentic.SimpleNamespace(rollout=rollout))
    tok = agentic.SimpleNamespace(pad_token_id=0, eos_token_id=2)

    async def two_samples():
        loops = [cls(cfg, None, tok, None, None, None) for _ in range(2)]  # one per sample, as verl does
        try:
            return await asyncio.gather(*(loop.run(VERL_SAMPLING, **ROW) for loop in loops))
        finally:
            for recorder in built:
                await recorder.aclose()

    outs = asyncio.run(two_samples())
    assert [o.extra_fields[OUTCOME_KEY] for o in outs] == ["ok", "ok"]
    assert len(built) == 1  # shared: one proxy, one HTTP server for the loop
    assert len(set(agent.endpoints)) == 2  # same task row, two recorder sessions
    loop_mod._recorders.clear()


def test_recorder_refuses_other_sampling_and_unknown_sessions(qwen):
    recorder = ProxyTokenRecorder(_proxy(qwen))

    async def go():
        with pytest.raises(HarnessConfigError, match="temperature"):
            await recorder.open("s1", {**VERL_SAMPLING, "temperature": 0.7})
        with pytest.raises(HarnessConfigError, match="not open"):
            await recorder.close("s2")
        await recorder.open("s3", VERL_SAMPLING)
        with pytest.raises(HarnessConfigError, match="already open"):
            await recorder.open("s3", VERL_SAMPLING)
        assert await recorder.close("s3") == []  # the agent never called: no turns
        await recorder.aclose()

    asyncio.run(go())


def test_estimator_other_than_yeto_grpo_is_refused(verl_loop):
    loop_mod, _ = verl_loop
    for estimator, ok in (("grpo", False), ("yeto_grpo", True)):
        cfg = agentic.SimpleNamespace(algorithm=agentic.SimpleNamespace(adv_estimator=estimator))
        if ok:
            loop_mod.check_estimator(cfg)
        else:
            with pytest.raises(HarnessConfigError, match="yeto_grpo"):
                loop_mod.check_estimator(cfg)
    loop_mod.check_estimator(agentic.SimpleNamespace())  # stub config: not checked


def test_turns_from_records_refuses_a_stream_no_turn_accounts_for():
    sample = {"segment": 0, "token_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 0],
              "turns": [{"turn": 0, "prompt_added": [1], "completion": [2], "logprobs": [-0.1],
                         "stop_added": []}]}
    from yeto.rl.agentic import TokenAlignmentError

    with pytest.raises(TokenAlignmentError, match="no turn accounts for"):
        turns_from_records({"sessions": [{"samples": [sample]}]})
    sample = {**sample, "token_ids": [1, 2], "loss_mask": [0, 1]}
    (turn,) = turns_from_records({"sessions": [{"samples": [sample]}]})
    assert (turn.prompt_ids, turn.completion_ids, turn.completion_logprobs) == ((1,), (2,), (-0.1,))


def test_a_trial_error_is_not_hidden_by_a_failing_recorder_close(verl_loop):
    loop_mod, _ = verl_loop

    class BrokenClose(agentic._Recorder):
        async def close(self, trajectory_id):
            raise RuntimeError("recorder gone")

    class Crash:
        async def run(self, task, endpoint):
            raise KeyboardInterrupt  # not an Exception: the outer layer lets it through

    loop = _make_loop(loop_mod, BrokenClose([]), Crash())
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(loop.run({}, **ROW))
