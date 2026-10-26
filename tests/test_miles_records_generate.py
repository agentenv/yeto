"""S25 N10.9: the Miles Harbor path with the records proxy as recorder (CPU).

Real pieces: ``records_generate.run_sample`` (outer layer, outcome rules,
``join_turns``, sample filling), ``ProxyTokenRecorder`` over HTTP, the gateway
and recording backend, ``SGLangRouterEngine`` (its HTTP call replaced by the
scripted model).  Fakes: the agent (Codex-like Responses loop), Miles' Sample.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from test_records_proxy import SCRIPT, SIGNED, ScriptedModel, _template
from test_verl_rollout_engine import IDENTITY, ExitOnErrorAgent, MalformedSecondReply
from test_verl_records_recorder import CodexLikeAgent

from yeto.rl.adapters.miles import records_generate as rg
from yeto.rl.agentic import OUTCOME_KEY
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

pytest.importorskip("aiohttp")

MILES_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 256}
ARGS = SimpleNamespace(rollout_max_response_len=4096, n_samples_per_prompt=4, grpo_std_normalization=True)


@pytest.fixture(scope="module")
def qwen():
    return _template("qwen3")


def _router_http(model: ScriptedModel, calls: list):
    def http(url: str, body: dict) -> dict:
        calls.append(body)
        ids = model(tuple(body["input_ids"]), 0)
        return {"text": "", "meta_info": {"output_token_logprobs": [[-0.5, t, None] for t in ids],
                                          "finish_reason": {"type": "stop"}}}
    return http


def _run(qwen, agent, model, monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "boat")
    calls: list = []
    engine = rg.SGLangRouterEngine("http://router", max_model_len=8192, identity=IDENTITY,
                                   http=_router_http(model, calls))
    recorder = ProxyTokenRecorder(RecordsProxy(engine, qwen, model="policy", signed_sampling=SIGNED, max_chains=1))
    sample = SimpleNamespace(metadata={"instance_id": "fix-git", "agent_name": "codex"}, tokens=[],
                             response_length=0, loss_mask=None, rollout_log_probs=None, status=None,
                             reward=None, remove_sample=False)

    async def go():
        try:
            return await rg.run_sample(sample, MILES_SAMPLING, ARGS, recorder=recorder, harness=agent)
        finally:
            await recorder.aclose()

    return asyncio.run(go()), calls


def test_sample_carries_the_tokens_the_router_sampled(qwen, monkeypatch):
    out, calls = _run(qwen, CodexLikeAgent(), ScriptedModel(qwen, copy.deepcopy(SCRIPT)), monkeypatch)
    assert out.metadata[OUTCOME_KEY] == "ok" and out.reward == 1.0 and out.status.name == "COMPLETED"
    assert len(calls) == 3 and calls[0]["return_logprob"] is True
    assert calls[0]["sampling_params"]["stop_token_ids"] == list(qwen.stop_token_ids)
    assert len(out.loss_mask) == len(out.rollout_log_probs) == out.response_length
    assert out.tokens[:len(calls[-1]["input_ids"])] == calls[-1]["input_ids"]  # trained == sampled
    assert all(lp == -0.5 for lp, m in zip(out.rollout_log_probs, out.loss_mask) if m)
    assert out.metadata["model_calls"] == 3 and out.metadata["truncated"] is False


def test_malformed_reply_is_a_trained_zero(qwen, monkeypatch):
    out, _ = _run(qwen, ExitOnErrorAgent(), MalformedSecondReply(qwen, copy.deepcopy(SCRIPT)), monkeypatch)
    assert out.metadata[OUTCOME_KEY] == "policy_limit" and out.reward == 0.0
    assert sum(out.loss_mask) > 0 and not out.remove_sample


def test_infrastructure_exit_is_aborted_and_never_trained(qwen, monkeypatch):
    out, _ = _run(qwen, ExitOnErrorAgent("EnvironmentStartTimeoutError"),
                  MalformedSecondReply(qwen, copy.deepcopy(SCRIPT)), monkeypatch)
    assert out.metadata[OUTCOME_KEY] == "aborted" and out.status.name == "ABORTED"
    assert out.remove_sample and out.loss_mask == [0] and out.reward == 0.0


def _s(reward, outcome="ok"):
    return SimpleNamespace(reward=reward, metadata={OUTCOME_KEY: outcome}, remove_sample=False)


def test_reward_post_process_leaves_aborted_out_of_the_group_baseline():
    samples = [_s(1.0), _s(0.0), _s(0.0, "aborted"), _s(0.0, "policy_limit"),  # group 1
               _s(0.0), _s(0.0), _s(0.0), _s(0.0)]  # group 2: no variance
    raw, adv = rg.reward_post_process(ARGS, samples)
    assert raw == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mean, std = 1 / 3, (((2 / 3) ** 2 + 2 * (1 / 3) ** 2) / 2) ** 0.5
    assert adv[0] == pytest.approx((1 - mean) / (std + 1e-6)) and adv[1] == pytest.approx(-mean / (std + 1e-6))
    assert adv[2] == 0.0 and samples[2].remove_sample and not samples[0].remove_sample
    assert adv[4:] == [0.0] * 4  # all-zero group: no gradient (expected for most TB2.1 groups)
    with pytest.raises(ValueError, match="whole groups"):
        rg.reward_post_process(ARGS, samples[:3])
