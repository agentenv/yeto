"""S28 G-120: agentic rounds of the engine sampler island (CPU).

Real pieces: ``build_engine_sampler`` with ``AgenticRound``, the outer layer
(``run_agent_task``, ``finish_sample``), ``ProxyTokenRecorder`` serving a
``RecordsProxy`` over HTTP on 127.0.0.1, ``FileSampleBuffer`` and the verl
buffer trainer's ``group_rows``.  Fakes: the engine (scripted ``FakeEngine``)
and the agent (a Codex-like harness calling the recorder URL over Responses).
Tokenizer: Qwen3-0.6B from the local HF cache (skip if absent); the nemotron3
template tests use the Nemotron-3.5-Super-VL-120B tokenizer files
(``YETO_NEMOTRON3_TOKENIZER`` or the dev machine's hf-meta copy).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_records_proxy import SCRIPT, ScriptedModel, _template, echo, responses_body, tool_output, user_item

from yeto.rl.agentic import TrialVerdict
from yeto.rl.engine.agentic_round import LOST_VERSION, AgenticRound
from yeto.rl.engine.buffer_payload import decode_group
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import SamplingParams
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

pytest.importorskip("aiohttp")

SIGNED = {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0}
TASK = {"instance_id": "fix-git", "agent_name": "codex", "trajectory_id": "row-traj"}


@pytest.fixture(scope="module")
def qwen():
    return _template("qwen3")


class Agent:
    """Codex-like harness: three Responses turns against the recorder URL; the
    verdict reward comes from ``rewards`` (one per trial, in call order)."""

    def __init__(self, rewards: list[float], *, fail: set[int] = frozenset(), turns: int = 3) -> None:
        self.rewards = list(rewards)
        self.fail = set(fail)
        self.turns = turns
        self.calls = 0
        self.endpoints: list[str] = []

    async def run(self, task: Any, endpoint: str) -> TrialVerdict:
        import aiohttp

        index = self.calls
        self.calls += 1
        self.endpoints.append(endpoint)
        if index in self.fail:
            raise RuntimeError("sandbox lost")  # infrastructure: ABORTED
        items = [user_item("read a.txt")]
        async with aiohttp.ClientSession() as http:
            for turn in range(self.turns):
                async with http.post(f"{endpoint}/responses", json=responses_body(copy.deepcopy(items))) as reply:
                    assert reply.status == 200, await reply.text()
                    response = await reply.json()
                items += echo(response["output"])
                items += [tool_output(o["call_id"], f"result {turn}")
                          for o in response["output"] if o["type"] == "function_call"]
        return TrialVerdict(rewards={"reward": self.rewards[index]})


def _island(tmp_path: Path, template, agent: Agent, *, n: int = 2, groups: int = 2, rounds: int = 1,
            min_samples: int = 2, script_turns: int = 3):
    from test_records_proxy import engine_for

    store = CheckpointStore(tmp_path)
    releases = PolicyReleaseStore(store)
    buffer = FileSampleBuffer(store, batch_size=1)
    engine = engine_for(ScriptedModel(template, copy.deepcopy(SCRIPT) * (n * groups * rounds)),
                        max_model_len=8192)
    proxy = RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, sampler_id="modal-0")
    recorder = ProxyTokenRecorder(proxy)
    events: list[tuple[str, dict]] = []
    config = EngineSamplerConfig(sampler_id="modal-0", num_rollouts=rounds, max_backlog_groups=100,
                                 sampling=SamplingParams(max_tokens=256, n=n), max_prompt_tokens=1024)

    def prompts(rollout_id: int) -> list[PromptItem]:
        return [PromptItem(f"row{rollout_id * groups + g}", f"TB2.1 task t{g}",
                           metadata={"row": g, "agent_task": dict(TASK)}) for g in range(groups)]

    def make(engine_, writer, **kw):
        return AgenticRound(engine_, writer, recorder=recorder, harness=agent, max_in_flight=1,
                            max_response_len=4096, max_prompt_len=4096, min_samples=min_samples,
                            task_defaults={"provider": "boat", "model_name": "policy"}, **kw)

    island = build_engine_sampler(config, engine=engine, releases=releases, buffer=buffer,
                                  emit=lambda name, **f: events.append((name, f)), prompts=prompts,
                                  reward=lambda item, c: 0.0, checks=PostLoadChecks(expected_expert_matrices=None,
                                                                                    require_fixed_answers=False),
                                  sleep=lambda s: None, make_round=make)
    return island, store, events, recorder, engine


def _entries(store):
    reader = FileSampleBuffer(store, batch_size=1)
    reader.refresh()
    return reader, [s.entry for s in reader.select("t", 100)]


def test_agentic_round_writes_groups_the_trainer_rows_read_with_the_tool_mask(tmp_path, qwen):
    from yeto.rl.adapters.verl import buffer_feed as bf

    agent = Agent([1.0, 0.0, 1.0, 1.0])
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent)
    try:
        assert island.run() == 1
    finally:
        island.round.close()
    reader, entries = _entries(store)
    assert len(entries) == 2 and {e.behavior_version for e in entries} == {engine.served_version()}
    samples = decode_group(reader.read_payload(entries[0].payload_ref))
    assert [s["reward"] for s in samples] == [1.0, 0.0]
    for s in samples:
        length = s["response_length"]
        assert len(s["rollout_log_probs"]) == len(s["loss_mask"]) == length and len(s["tokens"]) > length
        assert 0 < sum(s["loss_mask"]) < length  # tool output inside the response is masked
        assert all(lp == 0.0 for lp, m in zip(s["rollout_log_probs"], s["loss_mask"]) if m == 0)
        assert s["metadata"]["agent"]["yeto_agent_outcome"] == "ok" and s["metadata"]["agent_turns"] == 3
        assert s["metadata"]["task_id"] == "fix-git" and "agent_task" not in s["metadata"]
    # trained == sampled: the last engine prompt is a prefix of prompt + response
    last_prompt = list(engine.token_requests[-1][0][0])
    every = [x for e in entries for x in decode_group(reader.read_payload(e.payload_ref))]
    assert any(x["tokens"][:len(last_prompt)] == last_prompt for x in every)
    # the verl buffer trainer's rows carry the sample's mask, not all ones
    shape = bf.RowShape(prompt_length=4096, response_length=4096, pad_token_id=0)
    rows = bf.group_rows(samples, uid="u", param_versions=[0, 0], shape=shape)
    for s, mask in zip(samples, rows["tensors"]["response_mask"]):
        assert mask[:s["response_length"]] == s["loss_mask"] and sum(mask) == sum(s["loss_mask"])
    (round_event,) = [f for n, f in events if n == "rl_sampler_engine_round"]
    agentic = round_event["agentic"]
    assert agentic["trials"] == 4 and agentic["kept"] == 4 and agentic["lost_rate"] == 0.0
    assert agentic["reward_mean"] == 0.75 and round_event["groups"] == 2
    assert len(agent.endpoints) == 4 and all("/t/" in e for e in agent.endpoints)
    assert [n for n, _ in events][-1] == "rl_agentic_close"


def test_aborted_trials_are_left_out_counted_and_small_groups_dropped(tmp_path, qwen):
    agent = Agent([1.0] * 4, fail={0})  # trial 0 (group 0) loses its sandbox
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent, min_samples=2)
    try:
        assert island.run() == 1
    finally:
        island.round.close()
    _, entries = _entries(store)
    assert len(entries) == 1  # group 0 kept 1 < 2 trials: not written
    (round_event,) = [f for n, f in events if n == "rl_sampler_engine_round"]
    agentic = round_event["agentic"]
    assert agentic["trials"] == 4 and agentic["kept"] == 3 and agentic["lost"] == {"RuntimeError": 1}
    assert agentic["lost_rate"] == 0.25 and agentic["groups_below_min"] == 1
    assert agentic["outcomes"]["harness_outcome_aborted_total"] == 1


def test_trial_served_by_another_version_is_lost(tmp_path, qwen):
    from yeto.rl.agentic import TokenTurn

    class Recorder:
        def __init__(self) -> None:
            self.open_ids: list[str] = []

        async def open(self, trajectory_id, sampling_params):
            self.open_ids.append(trajectory_id)
            return "http://unused"

        async def close(self, trajectory_id):
            version = 7 if trajectory_id.endswith("-1") else None
            return [TokenTurn(prompt_ids=(1, 2), completion_ids=(3, 4), completion_logprobs=(-0.1, -0.2),
                              policy_version=version)]

    class Harness:
        async def run(self, task, endpoint):
            return TrialVerdict(rewards={"reward": 1.0})

    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.engine.buffer_payload import GroupWriter

    store = CheckpointStore(tmp_path)
    engine = FakeEngine()
    engine.start(None, 0)
    events = []
    rnd = AgenticRound(engine, GroupWriter(FileSampleBuffer(store, batch_size=1), "s"), recorder=Recorder(),
                       harness=Harness(), max_in_flight=4, max_response_len=64, min_samples=1,
                       task_defaults={"provider": "docker"},
                       prompts=lambda r: [PromptItem("p", "t", metadata={"agent_task": dict(TASK)})],
                       sampling=SamplingParams(max_tokens=8, n=2), provenance=_provenance(engine),
                       emit=lambda n, **f: events.append((n, f)))
    try:
        rnd(0)
    finally:
        rnd.close()
    agentic = [f for n, f in events if n == "rl_sampler_engine_round"][0]["agentic"]
    assert agentic["lost"] == {LOST_VERSION: 1} and agentic["kept"] == 1


def _provenance(engine):
    from yeto.rl.engine.sampler_engine import Provenance

    return Provenance.of(engine.identity, sampler_id="s", sampling=SamplingParams(max_tokens=8, n=2))


def test_a_row_without_an_agent_task_fails_the_round_loudly(tmp_path):
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.engine.buffer_payload import GroupWriter

    engine = FakeEngine()
    engine.start(None, 0)
    rnd = AgenticRound(engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"),
                       recorder=None, harness=None, max_in_flight=1, max_response_len=8,
                       prompts=lambda r: [PromptItem("p", "t")], sampling=SamplingParams(max_tokens=8, n=2),
                       provenance=_provenance(engine), emit=lambda n, **f: None, min_samples=1)
    try:
        with pytest.raises(HarnessConfigError, match="agent_task"):
            rnd(0)
    finally:
        rnd.loop.close()


def test_jsonl_prompts_carry_the_agent_task_and_the_spec_pairs_format_and_section(tmp_path):
    from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, build_prompts, build_reward

    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"prompt": "TB2.1 task fix-git", "metadata": TASK}) + "\n")
    spec = {"prompts": {"path": str(path), "input_key": "prompt", "groups_per_round": 1, "format": "agentic"},
            "agentic": {"template": "nemotron3"}}
    (item,) = build_prompts(spec)(0)
    assert item.metadata["agent_task"] == TASK
    assert build_reward(spec)(item, None) == 0.0
    with pytest.raises(SamplerSpecError, match="go together"):
        build_prompts({**spec, "agentic": None})


# ---------------------------------------------------------------------- nemotron3 template
NEMOTRON3_TOKENIZER = os.environ.get("YETO_NEMOTRON3_TOKENIZER",
                                     "/home/michael/work/hf-meta/nemotron35-super-vl-120b")


@pytest.fixture(scope="module")
def nemotron3():
    from yeto.rl.harness.chat_templates import ChatTemplate

    if not (Path(NEMOTRON3_TOKENIZER) / "tokenizer.json").is_file():
        pytest.skip("Nemotron-3 tokenizer files not available")
    return ChatTemplate.load("nemotron3", tokenizer_path=NEMOTRON3_TOKENIZER)


def test_nemotron3_template_aligns_merges_codex_system_messages_and_keeps_history_reasoning(nemotron3):
    from yeto.rl.harness.chat_templates import check_alignment
    from yeto.rl.harness.gateway.proxy import chat_tools
    from test_records_proxy import TOOLS, assistant, call

    tools = chat_tools("responses", {"tools": TOOLS})
    messages = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                {"role": "user", "content": "u"}]
    text = nemotron3.render(messages, tools=tools, add_generation_prompt=True)
    assert text.count("<|im_start|>system") == 1 and text.endswith("<think>\n")
    assert check_alignment(nemotron3, messages, tools=tools).aligned
    history = [*messages, assistant("first thought", calls=[call("c1", "ls")]),
               {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
               {"role": "user", "content": "again"}]
    rendered = nemotron3.render(history, tools=tools, add_generation_prompt=True)
    assert "first thought" in rendered  # truncate_history_thinking=False: the stream stays append-only
    assert check_alignment(nemotron3, history, tools=tools).aligned


def test_nemotron3_codex_loop_records_one_aligned_stream(nemotron3):
    from test_records_proxy import engine_for, run_codex

    engine = engine_for(ScriptedModel(nemotron3, copy.deepcopy(SCRIPT)))
    proxy = RecordsProxy(engine, nemotron3, model="policy", signed_sampling=SIGNED, max_chains=1)
    outputs = run_codex(proxy, "t1")
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t1")
    assert records["gateway"]["invalidated"] is None
    assert records["counters"].get("render_drift_turns", 0) == 0
    (sample,) = records["sessions"][0]["samples"]
    assert [t["kind"] for t in sample["turns"]] == ["action"] * 3
