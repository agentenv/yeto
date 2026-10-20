"""S25 N10.9: the records proxy over verl's own rollout client, behind a path secret (CPU).

Real pieces: ``HarborAgentLoop.run``, ``ProxyTokenRecorder`` serving a
``RecordsProxy`` over HTTP, the gateway and recording backend, and
``VerlRolloutEngine``.  Fakes: verl's agent-loop module (stub), verl's
``LLMServerClient`` (an async ``generate`` that replies with the scripted
model's tokens), the agent (Codex-like Responses loop).  Tokenizer: Qwen3-0.6B
from the local HF cache (skip if absent).
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

import pytest
import test_agentic_harbor as agentic
from test_agentic_harbor import _make_loop
from test_records_proxy import SCRIPT, SIGNED, ScriptedModel, _template
from test_verl_records_recorder import ROW, VERL_SAMPLING, CodexLikeAgent

from yeto.rl.agentic import OUTCOME_KEY, HarnessConfigError
from yeto.rl.engine.sampler_engine import FINISH_LENGTH, FINISH_STOP, EngineIdentity, SamplerEngineError
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

pytest.importorskip("aiohttp")

verl_loop = agentic.verl_loop  # the verl agent-loop stub fixture
SECRET = "s" * 32
IDENTITY = EngineIdentity(engine="verl-rollout", engine_version="test", device_family="cuda",
                          card_model="H100", precision="bf16", quant_scheme="bf16", topology="unknown")


@pytest.fixture(scope="module")
def qwen():
    return _template("qwen3")


class FakeVerlClient:
    """verl ``LLMServerClient`` stand-in: replies with the scripted model's tokens."""

    def __init__(self, model: ScriptedModel, *, aborted: bool = False) -> None:
        self.model = model
        self.aborted = aborted
        self.calls: list[dict[str, Any]] = []
        self.loops: set[int] = set()

    async def generate(self, request_id: str, *, prompt_ids: list[int], sampling_params: dict[str, Any]) -> Any:
        self.loops.add(id(asyncio.get_running_loop()))
        self.calls.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        if self.aborted:
            return SimpleNamespace(token_ids=[], log_probs=None, stop_reason="aborted")
        ids = self.model(tuple(prompt_ids), 0)
        return SimpleNamespace(token_ids=ids, log_probs=[-0.25] * len(ids), stop_reason="completed",
                               extra_fields={"global_steps": 3})


def _verl_proxy(template) -> tuple[RecordsProxy, Any]:
    from yeto.rl.adapters.verl.rollout_engine import VerlRolloutEngine

    engine = VerlRolloutEngine(max_model_len=8192, identity=IDENTITY)
    return RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, max_chains=1), engine


def test_agent_loop_samples_through_verls_client_behind_the_path_secret(verl_loop, qwen):
    loop_mod, _ = verl_loop
    proxy, engine = _verl_proxy(qwen)
    recorder = ProxyTokenRecorder(proxy, path_secret=SECRET)
    agent = CodexLikeAgent()
    loop = _make_loop(loop_mod, recorder, agent, prompt_length=4096, response_length=4096)
    client = FakeVerlClient(ScriptedModel(qwen, copy.deepcopy(SCRIPT)))
    loop.server_manager = client  # verl's AgentLoopBase sets it; the stub does not

    async def go():
        try:
            out = await loop.run(VERL_SAMPLING, **ROW)
            import aiohttp

            base = recorder._base_url.split("/k/")[0]
            async with aiohttp.ClientSession() as http:  # the root is not served, only the secret prefix
                async with http.get(f"{base}/t/traj-1/records") as reply:
                    root_status = reply.status
        finally:
            await recorder.aclose()
        return out, root_status

    out, root_status = asyncio.run(go())
    assert out.extra_fields[OUTCOME_KEY] == "ok" and out.reward_score == 1.0 and out.num_turns == 6
    (endpoint,) = agent.endpoints
    assert f"/k/{SECRET}/t/traj-1-" in endpoint and root_status == 404
    assert engine.bound and len(client.calls) == 3 and len(client.loops) == 1
    params = client.calls[0]["sampling_params"]
    assert params["logprobs"] is True and params["temperature"] == 1.0 and params["top_k"] == -1
    assert params["stop_token_ids"] == sorted(qwen.stop_token_ids) and params["max_tokens"] == 256
    # trained == sampled: the last prompt verl got is a prefix of prompt + response
    last_prompt = client.calls[-1]["prompt_ids"]
    assert out.prompt_ids + out.response_ids[:len(last_prompt) - len(out.prompt_ids)] == last_prompt
    trained = [lp for lp, m in zip(out.response_logprobs, out.response_mask) if m == 1]
    assert trained and all(lp == -0.25 for lp in trained)
    assert out.extra_fields["provenance"]["engine"] == "verl-rollout"
    assert out.extra_fields["length_capped_turns"] == 0
    # verl V1 needs the weights' step range per sample
    assert out.extra_fields["min_global_steps"] == out.extra_fields["max_global_steps"] == 3
    assert engine.pop_steps(endpoint.split("/t/")[1].split("/")[0]) == []  # taken by the loop


def test_an_aborted_verl_request_aborts_the_sample(verl_loop, qwen):
    loop_mod, _ = verl_loop
    proxy, _ = _verl_proxy(qwen)
    recorder = ProxyTokenRecorder(proxy)
    agent = CodexLikeAgent()
    loop = _make_loop(loop_mod, recorder, agent, prompt_length=4096, response_length=4096)
    loop.server_manager = FakeVerlClient(ScriptedModel(qwen, copy.deepcopy(SCRIPT)), aborted=True)

    async def go():
        try:
            return await loop.run(VERL_SAMPLING, **ROW)
        finally:
            await recorder.aclose()

    out = asyncio.run(go())
    assert out.extra_fields[OUTCOME_KEY] == "aborted" and sum(out.response_mask) == 0
    assert out.extra_fields["min_global_steps"] == out.extra_fields["max_global_steps"] == 0  # no reply seen


def test_unbound_engine_and_short_secret_fail_closed(qwen):
    from yeto.rl.adapters.verl.rollout_engine import VerlRolloutEngine, finish_reason
    from yeto.rl.engine.sampler_engine import SamplingParams

    engine = VerlRolloutEngine(max_model_len=64, identity=IDENTITY)
    with pytest.raises(SamplerEngineError, match="not bound"):
        engine.generate_tokens([[1, 2]], SamplingParams(max_tokens=4))
    with pytest.raises(HarnessConfigError, match="at least"):
        ProxyTokenRecorder(_verl_proxy(qwen)[0], path_secret="short")
    assert finish_reason((5, 7), {7}, 2) == FINISH_STOP
    assert finish_reason((5, 6), {7}, 2) == FINISH_LENGTH
    assert finish_reason((5,), {7}, 2) == FINISH_STOP


class ExitOnErrorAgent(CodexLikeAgent):
    """Codex on a 400: the turn fails and the agent process exits non-zero (Harbor's verdict)."""

    def __init__(self, exception_type: str = "NonZeroAgentExitCodeError") -> None:
        super().__init__()
        self.exception_type = exception_type
        self.statuses: list[int] = []

    async def run(self, task: Any, endpoint: str) -> Any:
        import aiohttp

        from test_records_proxy import echo, responses_body, tool_output, user_item

        from yeto.rl.agentic import TrialVerdict

        self.endpoints.append(endpoint)
        items = [user_item("read a.txt")]
        async with aiohttp.ClientSession() as http:
            for turn in range(3):
                async with http.post(f"{endpoint}/responses", json=responses_body(copy.deepcopy(items))) as reply:
                    self.statuses.append(reply.status)
                    if reply.status != 200:
                        return TrialVerdict(rewards=None, exception_type=self.exception_type)
                    response = await reply.json()
                items += echo(response["output"])
                items += [tool_output(o["call_id"], f"result {turn}")
                          for o in response["output"] if o["type"] == "function_call"]
        return TrialVerdict(rewards={"reward": 1.0})


class MalformedSecondReply(ScriptedModel):
    """The second reply is a tool call that is not JSON (as Qwen3-4B wrote with heredocs)."""

    def __call__(self, prompt: tuple[int, ...], index: int) -> list[int]:
        if len(self.replies) == len(SCRIPT) - 1:
            self.replies.pop(0)
            return self.template.encode('<tool_call>\n{"name": "shell", "arguments": {"cmd": "cat > a\nb"}\n</tool_call>') + \
                [self.template.encode("<|im_end|>")[0]]
        return super().__call__(prompt, index)


@pytest.mark.parametrize("exception_type,outcome", [("NonZeroAgentExitCodeError", "policy_limit"),
                                                    ("EnvironmentStartTimeoutError", "aborted")])
def test_malformed_tool_call_is_a_trained_zero_not_an_abort(verl_loop, qwen, exception_type, outcome):
    loop_mod, _ = verl_loop
    proxy, _ = _verl_proxy(qwen)
    recorder = ProxyTokenRecorder(proxy)
    agent = ExitOnErrorAgent(exception_type)
    loop = _make_loop(loop_mod, recorder, agent, prompt_length=4096, response_length=4096)
    loop.server_manager = FakeVerlClient(MalformedSecondReply(qwen, copy.deepcopy(SCRIPT)))

    async def go():
        try:
            return await loop.run(VERL_SAMPLING, **ROW)
        finally:
            await recorder.aclose()

    out = asyncio.run(go())
    assert agent.statuses == [200, 400]
    assert out.extra_fields[OUTCOME_KEY] == outcome and out.reward_score == 0.0
    if outcome == "policy_limit":
        # the malformed reply was sampled and is trained (mask 1) as part of a 0-reward sample
        assert out.extra_fields["outcome_reason"].startswith("policy_output_error")
        assert sum(out.response_mask) > 0
    else:
        assert sum(out.response_mask) == 0  # another exit cause stays an infrastructure abort
