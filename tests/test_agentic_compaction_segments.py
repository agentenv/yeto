"""CompactionRL segments on the engine sampler's agentic path (CPU).

A records proxy with a ``compaction`` config compacts a trial's context; the
agentic round writes one buffer sample per compaction segment into the trial's
group (``agentic_samples``). Real pieces: ``RecordsProxy`` (records backend
with compaction), ``ProxyTokenRecorder`` over HTTP on 127.0.0.1,
``finish_sample``, ``AgenticRound``, ``GroupWriter`` / ``FileSampleBuffer``.
Fakes: the engine (``FakeEngine`` with a scripted model that answers the
summary request with a summary) and a Codex-like agent (Responses over HTTP).
Tokenizer: GLM-4.7-Flash from the local HF cache (GLM keeps every turn's
reasoning, so the summary request extends the stream; tests/hf_assets.py).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

import numpy
import pytest
from test_records_proxy import (_template, assistant, call, echo, model_text, responses_body, tool_output, turn_end,
                                user_item)

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.agentic import AgentTask, AgentTrajectory, HarnessConfigError, Outcome, TokenTurn, TrialVerdict
from yeto.rl.agentic.admission import HarnessMetrics
from yeto.rl.agentic.finish import CompactionSegment, finish_sample
from yeto.rl.compaction import RESUME_TEMPLATE, CompactionConfig
from yeto.rl.engine.agentic_round import GROUP_EVENT, AgenticRound, agentic_sample, agentic_samples
from yeto.rl.engine.buffer_payload import (ROLLOUT_ROUTED_EXPERTS, ROLLOUT_TOP_LOGPROBS, ROLLOUT_TOP_LOGPROBS_IDS,
                                           GroupWriter, decode_group, encode_group, miles_sample_dict)
from yeto.rl.engine.engine_sampler import PromptItem
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_engine import Provenance, SamplingParams
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import (ProxyTokenRecorder, recorder_from_env, segments_from_records,
                                              turns_from_records)

pytest.importorskip("aiohttp")

SIGNED = {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0}
TASK = {"instance_id": "fix-git", "agent_name": "codex"}
SUMMARY_END = "Do not call any tool in this turn."  # the last sentence of SUMMARY_PROMPT
LAST = "LAST"  # a tool output that ends with it asks the model to finish
EXPERTS, LAYERS, TOPK, K = 16, 3, 2, 4
# Budget calibrated on GLM-4.7-Flash: each step adds ~140 tokens, the first prompt
# is ~164 tokens and a compacted segment's first prompt ~337: with t_comp 1 the
# third request compacts, and every second one after it.
BUDGET = 350
RESUME_HEAD = RESUME_TEMPLATE.split("{summary}")[0].strip()


@pytest.fixture(scope="module")
def glm():
    return _template("glm5")


class Model:
    """``token_reply`` of the fake engine: a summary for the summary request, a
    final answer after a tool output that ends with :data:`LAST`, else a tool call."""

    def __init__(self, template, *, on_summary=None) -> None:
        self.template = template
        self.on_summary = on_summary

    def __call__(self, prompt: tuple[int, ...], index: int) -> list[int]:
        tail = self.template.decode(list(prompt[-30:]))
        if SUMMARY_END in tail:
            message = assistant("summarize", content="<summary>1. ran the steps so far</summary>")
            if self.on_summary is not None:
                self.on_summary()
        elif LAST in tail:
            message = assistant("finish", content="All done.")
        else:
            message = assistant("next step", calls=[call("c", "run")])
        reply = self.template.encode(model_text(self.template, message))
        return reply + [self.template.encode(turn_end(self.template, message))[0]]


class Engine(FakeEngine):
    """The fake engine; ``live``: it updates during a generation and stamps each
    completion with the version it was sampled at. Keeps every completion."""

    def __init__(self, *, live: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.updates_during_generation = live
        self.completions: list[tuple[tuple[int, ...], Any]] = []

    def generate_tokens(self, prompts, sampling, *, stop_token_ids=()):
        version = self.served_version()
        out = super().generate_tokens(prompts, sampling, stop_token_ids=stop_token_ids)
        if self.updates_during_generation:
            out = [[dataclasses.replace(c, weight_version=version) for c in group] for group in out]
        for prompt, group in zip(prompts, out):
            self.completions.append((tuple(prompt), group[0]))
        return out


def _engine(template, *, max_model_len: int = 8192, arrays: bool = False, live: bool = False,
            on_summary=None) -> Engine:
    kw = {"routed_experts": (EXPERTS, LAYERS, TOPK), "top_logprobs_num": K} if arrays else {}
    engine = Engine(token_reply=Model(template, on_summary=on_summary), max_model_len=max_model_len, live=live, **kw)
    engine.start(None, 3)
    return engine


def _proxy(engine, template, compaction: CompactionConfig | None, *, r3_response_len: int | None = None):
    return RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, sampler_id="s",
                        compaction=compaction, r3_response_len=r3_response_len)


def _cfg(max_compactions: int = 3, budget: int = BUDGET) -> CompactionConfig:
    return CompactionConfig(context_budget=budget, t_comp=1, keep_recent_steps=1, max_compactions=max_compactions)


def _tool_text(turn: int, last: bool) -> str:
    return " ".join(f"line{turn}-{k}" for k in range(30)) + (f" {LAST}" if last else "")


class Agent:
    """Codex-like harness: ``steps`` Responses calls (the last one after a
    :data:`LAST` tool output). A refused call (HTTP 409, the context is full)
    ends the agent as a non-zero exit, as Harbor reports it."""

    def __init__(self, steps: int, rewards: list[float] | None = None) -> None:
        self.steps = steps
        self.rewards = list(rewards or [])
        self.calls = 0

    async def run(self, task: Any, endpoint: str) -> TrialVerdict:
        import aiohttp

        reward = self.rewards[self.calls] if self.calls < len(self.rewards) else 1.0
        self.calls += 1
        items = [user_item("do the task")]
        async with aiohttp.ClientSession() as http:
            for turn in range(self.steps):
                async with http.post(f"{endpoint}/responses", json=responses_body(copy.deepcopy(items))) as reply:
                    if reply.status != 200:
                        return TrialVerdict(rewards=None, exception_type="NonZeroAgentExitCodeError",
                                            exception_message=(await reply.text())[:200])
                    response = await reply.json()
                items += echo(response["output"])
                items += [tool_output(o["call_id"], _tool_text(turn, turn == self.steps - 2))
                          for o in response["output"] if o["type"] == "function_call"]
        return TrialVerdict(rewards={"reward": reward})


def _trajectory(session: str, verdict: TrialVerdict) -> AgentTrajectory:
    task = AgentTask(task_id="fix-git", agent="codex", provider="docker", trajectory_id=session)
    if verdict.exception_type is not None:
        return AgentTrajectory(task=task, outcome=Outcome.ABORTED, reward=0.0, reason=verdict.exception_type,
                               verdict=verdict)
    return AgentTrajectory(task=task, outcome=Outcome.OK, reward=float(verdict.rewards["reward"]),
                           reason="verified", verdict=verdict)


def _finished(recorder: ProxyTokenRecorder, steps: int, *, segmented: bool, session: str = "t-0",
              max_response_len: int = 4096, max_prompt_len: int | None = None):
    """One trial through the recorder over HTTP, then ``finish_sample``."""
    async def go():
        try:
            endpoint = await recorder.open(session, {})
            verdict = await Agent(steps).run(None, endpoint)
            return await finish_sample(_trajectory(session, verdict), recorder, session, HarnessMetrics(),
                                       max_response_len=max_response_len, max_prompt_len=max_prompt_len,
                                       segmented=segmented)
        finally:
            await recorder.aclose()

    return asyncio.run(go())


ITEM = PromptItem("p0", "task", metadata={"agent_task": dict(TASK), "row": 0})


def _round(tmp_path: Path, recorder, engine, agent, *, n: int = 2, live: bool = False,
           max_response_len: int = 4096, max_prompt_len: int | None = None):
    """One group (``n`` trials of one task) through the agentic round; its samples and events."""
    sampling = SamplingParams(max_tokens=256, n=n)
    store = CheckpointStore(tmp_path)
    events: list[tuple[str, dict]] = []
    rnd = AgenticRound(engine, GroupWriter(FileSampleBuffer(store, batch_size=1), "s"), recorder=recorder,
                       harness=agent, max_in_flight=1, max_response_len=max_response_len,
                       max_prompt_len=max_prompt_len, min_samples=1, task_defaults={"provider": "docker"},
                       prompts=lambda rid: [ITEM] if rid == 0 else [], sampling=sampling,
                       provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling),
                       emit=lambda name, **f: events.append((name, f)),
                       data_cursor=lambda rid: {"row_offset": int(rid) + 1},
                       weight_switch="live" if live else "drain", release_poll_s=0.0, wait_s=0.005)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    reader = FileSampleBuffer(store, batch_size=1)
    reader.refresh()
    entries = [selected.entry for selected in reader.select("t", 100)]
    samples = [s for e in entries for s in decode_group(reader.read_payload(e.payload_ref))]
    return samples, events


def _by_trial(samples: list[dict]) -> dict[int, list[dict]]:
    trials: dict[int, list[dict]] = {}
    for sample in samples:
        trials.setdefault(sample["index"], []).append(sample)
    return trials


def _check_segments(trial: list[dict], engine: Engine, cfg: CompactionConfig, template) -> None:
    """Contract section 2 on the samples of one compacted trial."""
    count = len(trial)
    metas = [s["metadata"]["compaction"] for s in trial]
    assert [m["segment_index"] for m in metas] == list(range(count))
    assert {m["num_segments"] for m in metas} == {count} and {m["schema"] for m in metas} == {1}
    assert {m["trajectory_id"] for m in metas} == {trial[0]["metadata"]["trajectory_id"]}
    assert {m["context_budget"] for m in metas} == {cfg.context_budget}
    assert len({s["group_index"] for s in trial}) == 1 and len({s["reward"] for s in trial}) == 1
    # every segment names the trial's task by its prompt (reward evidence hashes it); its own
    # stream (resume prompt included) is in ``tokens``
    assert {s["prompt"] for s in trial} == {ITEM.prompt} and {s["label"] for s in trial} == {ITEM.label}
    trained = [int(sum(s["loss_mask"])) for s in trial]
    assert [m["segment_tokens"] for m in metas] == trained
    assert [m["tokens_after"] for m in metas] == [sum(trained[i + 1:]) for i in range(count)]
    assert [m["summary_ok"] for m in metas] == [True] * (count - 1) + [None]
    assert metas[0]["kept_steps"] is None and all(m["kept_steps"] in (0, 1) for m in metas[1:])
    assert all(s["status"] == "completed" for s in trial[:-1])
    prompts = {p for p, _ in engine.completions}
    for number, sample in enumerate(trial):
        length = sample["response_length"]
        assert len(sample["loss_mask"]) == len(sample["rollout_log_probs"]) == length > 0
        assert all(lp == 0.0 for lp, m in zip(sample["rollout_log_probs"], sample["loss_mask"]) if m == 0)
        prompt = tuple(sample["tokens"][:-length])
        assert prompt in prompts  # each segment's prompt is what the engine saw first in it
        text = template.decode(list(prompt))
        assert (RESUME_HEAD in text) == (number > 0)
        # trained == sampled: the segment's last call ends the segment's stream
        last = [c for p, c in engine.completions if p + tuple(c.token_ids) == tuple(sample["tokens"])]
        assert last, f"segment {number} is not the stream of one engine call"
        assert sample["metadata"]["agent_turns"] >= 1


# ---------------------------------------------------------------------- records -> segments
def test_segments_from_records_groups_the_calls_per_compaction_segment(glm):
    engine = _engine(glm)
    proxy = _proxy(engine, glm, _cfg())
    recorder = ProxyTokenRecorder(proxy, allow_compaction=True)
    session = "t-0"

    async def go():
        try:
            endpoint = await recorder.open(session, {})
            await Agent(5).run(None, endpoint)
            return proxy.records(session)
        finally:
            await recorder.aclose()

    records = asyncio.run(go())
    assert records["counters"]["compactions"] == 3
    groups = segments_from_records(records, context_budget=BUDGET)
    assert len(groups) == 4 and {g.context_budget for g in groups} == {BUDGET}
    assert [t for g in groups for t in g.turns] == turns_from_records(records)  # the flat calls, unchanged
    assert [g.kept_steps for g in groups] == [None, 1, 1, 1] and [g.summary_ok for g in groups] == [True] * 3 + [None]
    for group in groups:  # each group extends its own first prompt
        first = group.turns[0].prompt_ids
        assert all(t.prompt_ids[:len(first)] == first for t in group.turns)


def _sample(segment: int, ids: list[int], completion_at: int) -> dict:
    """A records sample whose one call completes ``ids[completion_at:]``."""
    return {"segment": segment, "token_ids": ids, "loss_mask": [0] * completion_at + [1] * (len(ids) - completion_at),
            "logprobs": [0.0] * len(ids), "kept_steps": 1 if segment else None, "summary_ok": None,
            "turns": [{"turn": 0, "prompt_added": ids[:completion_at], "completion": ids[completion_at:],
                       "logprobs": [-0.1] * (len(ids) - completion_at)}]}


def test_a_later_session_continues_the_segment_and_an_unsampled_segment_is_dropped():
    records = {"sessions": [{"samples": [_sample(0, [1, 2, 3], 2), _sample(1, [7, 8, 9], 2),
                                         {**_sample(2, [], 0), "turns": []}]},
                            {"samples": [_sample(0, [1, 2, 3, 4], 3)]}]}
    groups = segments_from_records(records)
    assert [[t.completion_ids for t in g.turns] for g in groups] == [[(3,)], [(9,), (4,)]]
    assert [t for g in groups for t in g.turns] == turns_from_records(records)


# ---------------------------------------------------------------------- no compaction: main's bytes
def test_without_compaction_the_samples_are_the_unsegmented_samples_byte_for_byte(glm):
    """The same trial on three proxies: no compaction (main's path: ``close`` +
    ``agentic_sample``), the segmented path, and a compaction budget it never
    reaches. Engine arrays (top-K, R3) on, cut to the training window."""
    def run(compaction, segmented, allow):
        engine = _engine(glm, arrays=True)
        proxy = _proxy(engine, glm, compaction, r3_response_len=4096)
        return _finished(ProxyTokenRecorder(proxy, allow_compaction=allow), 4, segmented=segmented)

    main = run(None, False, False)
    assert main.segments is not None and main.parts == ()
    reference = [agentic_sample(ITEM, main, version=3, provenance={"engine": "fake"}, group_index=0, index=1)]
    for compaction, allow in ((None, False), (None, True), (_cfg(budget=100_000), True)):
        new = run(compaction, True, allow)
        assert new.parts == () and new.turns == main.turns
        for versions in (None, (3, 3)):  # drain, live
            expected = [agentic_sample(ITEM, main, version=3, provenance={"engine": "fake"}, group_index=0,
                                       index=1, versions=versions)]
            got = agentic_samples(ITEM, new, version=3, provenance={"engine": "fake"}, group_index=0, index=1,
                                  versions=versions, start_version=3)
            assert encode_group(got) == encode_group(expected)
        assert "compaction" not in got[0]["metadata"]
    assert reference[0][ROLLOUT_ROUTED_EXPERTS].shape[0] == len(reference[0]["tokens"]) - 1


def test_a_round_without_compaction_writes_one_sample_per_trial(tmp_path, glm):
    engine = _engine(glm)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, _cfg(budget=100_000)), allow_compaction=True)
    samples, events = _round(tmp_path, recorder, engine, Agent(4, [1.0, 0.0]))
    assert [s["index"] for s in samples] == [0, 1] and [s["reward"] for s in samples] == [1.0, 0.0]
    assert all("compaction" not in s["metadata"] and s["status"] == "completed" for s in samples)
    (round_event,) = [f for name, f in events if name == "rl_sampler_engine_round"]
    assert "compaction_no_room" not in round_event["agentic"]  # the round event keys of main


# ---------------------------------------------------------------------- compacted trials
@pytest.mark.parametrize("steps,compactions", [(3, 1), (5, 3)])
def test_a_compacted_trial_writes_one_sample_per_segment_into_its_group(tmp_path, glm, steps, compactions):
    engine = _engine(glm)
    cfg = _cfg()
    recorder = ProxyTokenRecorder(_proxy(engine, glm, cfg), allow_compaction=True)
    samples, events = _round(tmp_path, recorder, engine, Agent(steps, [1.0, 0.0]))
    trials = _by_trial(samples)
    assert sorted(trials) == [0, 1] and len(samples) == 2 * (compactions + 1)
    assert [trials[0][0]["reward"], trials[1][0]["reward"]] == [1.0, 0.0]
    for trial in trials.values():
        assert len(trial) == compactions + 1
        _check_segments(trial, engine, cfg, glm)
        assert trial[-1]["status"] == "completed"
        assert {s["metadata"]["weight_version"] for s in trial} == {"3"}
    (group,) = [f for name, f in events if name == GROUP_EVENT]
    assert group["kept"] == 2 and group["trials"] == 2 and group["written"]


def test_the_context_full_after_the_last_compaction_truncates_only_the_last_segment(tmp_path, glm):
    """After its last compaction the trial runs until its context reaches
    context_budget (not the engine's larger max_model_len): the last segment
    ends there, truncated."""
    engine = _engine(glm, max_model_len=900)
    cfg = _cfg(max_compactions=1)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, cfg), allow_compaction=True)
    samples, _ = _round(tmp_path, recorder, engine, Agent(12), n=1)
    trial = _by_trial(samples)[0]
    assert len(trial) == 2
    _check_segments(trial, engine, cfg, glm)
    assert [s["status"] for s in trial] == ["completed", "truncated"]
    assert trial[-1]["metadata"]["finish_reason"] == "length"
    assert {s["reward"] for s in trial} == {0.0}  # the policy filled its context: a policy limit, trained as 0
    assert {s["metadata"]["agent"]["yeto_agent_outcome"] for s in trial} == {Outcome.POLICY_LIMIT.value}
    # the next prompt (this stream + a ~140-token tool output) reached context_budget, far below max_model_len
    assert cfg.context_budget - 200 < len(trial[-1]["tokens"]) <= cfg.context_budget < 900


def test_no_room_for_the_summary_request_ends_the_trial_truncated_even_with_one_segment(tmp_path, glm):
    """The third request triggers compaction (stream ~444 > budget 350), but the
    summary request does not fit max_model_len 420, not even with the newest tool
    output left out of it (~450 tokens): the trial ends there, one segment,
    truncated, reward 0, counted as compaction_no_room."""
    from yeto.rl.engine import agentic_round
    from yeto.rl.harness.gateway import records

    assert agentic_round.COMPACTION_NO_ROOM == records.COMPACTION_NO_ROOM
    engine = _engine(glm, max_model_len=420)  # 380-450 all end this way on GLM-4.7-Flash
    recorder = ProxyTokenRecorder(_proxy(engine, glm, _cfg()), allow_compaction=True)
    samples, events = _round(tmp_path, recorder, engine, Agent(6), n=1)
    (sample,) = samples
    assert sample["status"] == "truncated" and sample["metadata"]["finish_reason"] == "length"
    assert sample["reward"] == 0.0 and "compaction" not in sample["metadata"]  # never compacted: one stream
    assert sample["metadata"]["agent"]["yeto_agent_outcome"] == Outcome.POLICY_LIMIT.value
    (round_event,) = [f for name, f in events if name == "rl_sampler_engine_round"]
    assert round_event["agentic"]["compaction_no_room"] == 1


def test_no_room_for_a_later_summary_truncates_the_last_segment(tmp_path, glm):
    """Compacted once; the next summary request does not fit: the last segment is truncated.

    The first summary request fits with the newest tool output left out of it
    (~510 tokens); the second segment's one is longer (it starts from the resume
    message and the kept step, ~520): max_model_len 475-490 on GLM-4.7-Flash
    lets the first through and not the second."""
    engine = _engine(glm, max_model_len=485)
    cfg = _cfg()
    recorder = ProxyTokenRecorder(_proxy(engine, glm, cfg), allow_compaction=True)
    samples, events = _round(tmp_path, recorder, engine, Agent(8), n=1)
    trial = _by_trial(samples)[0]
    assert len(trial) == 2
    _check_segments(trial, engine, cfg, glm)
    assert trial[-1]["status"] == "truncated" and {s["reward"] for s in trial} == {0.0}
    (round_event,) = [f for name, f in events if name == "rl_sampler_engine_round"]
    assert round_event["agentic"]["compaction_no_room"] == 1


def test_r3_rows_and_topk_are_each_segments_own(tmp_path, glm):
    """Each segment's R3 rows and top-K are its own: the rows of its last call
    over its own tokens (its R3 window starts at its own first prompt), the
    top-K of its own response. With segments bounded by context_budget a later
    segment no longer outgrows the trial's first prompt + max_response_len (the
    window that would have cut it); records.RecordingBackend._r3_rows_cap keeps
    the per-segment window."""
    engine = _engine(glm, arrays=True)
    cfg = _cfg(max_compactions=1)
    max_response_len = 450  # segment 1: prompt ~337 + response ~426 > first prompt ~164 + 450
    recorder = ProxyTokenRecorder(_proxy(engine, glm, cfg, r3_response_len=max_response_len), allow_compaction=True)
    samples, _ = _round(tmp_path, recorder, engine, Agent(6), n=1, max_response_len=max_response_len)
    trial = _by_trial(samples)[0]
    assert len(trial) == 2
    _check_segments(trial, engine, cfg, glm)
    first_prompt = len(trial[0]["tokens"]) - trial[0]["response_length"]
    assert len(trial[1]["tokens"]) - trial[1]["response_length"] > first_prompt  # its own, longer prompt
    for sample in trial:
        tokens, length = sample["tokens"], sample["response_length"]
        rows = numpy.asarray(sample[ROLLOUT_ROUTED_EXPERTS])
        assert rows.shape == (len(tokens) - 1, LAYERS, TOPK)
        (last,) = [c for p, c in engine.completions if p + tuple(c.token_ids) == tuple(tokens)]
        assert numpy.array_equal(rows, numpy.asarray(last.routed_experts)[:len(tokens) - 1])
        top_ids, top_vals = numpy.asarray(sample[ROLLOUT_TOP_LOGPROBS_IDS]), numpy.asarray(sample[ROLLOUT_TOP_LOGPROBS])
        assert top_ids.shape == top_vals.shape == (length, K)
        mask = numpy.asarray(sample["loss_mask"], dtype=bool)
        response = numpy.asarray(tokens[-length:])
        assert (top_ids[mask, 0] == response[mask]).all() and (top_ids[~mask] == -1).all()
        n = len(last.token_ids)  # the segment's last call sits at the end of its response
        assert numpy.array_equal(top_ids[-n:], last.top_logprobs_ids)
        assert numpy.array_equal(top_vals[-n:], numpy.asarray(last.top_logprobs, dtype=top_vals.dtype))
        miles = miles_sample_dict(sample, r3_num_experts=EXPERTS)  # the trainer's conversion takes each segment
        assert len(miles["tokens"]) == len(tokens)


def test_a_reload_between_segments_gives_each_segment_its_own_versions(tmp_path, glm):
    """Live weight switch: the engine loads v4 right after the first summary, so
    the first segment is all v3 and the second all v4."""
    holder: dict[str, Engine] = {}

    def reload() -> None:
        if holder["engine"].served_version() == 3:
            holder["engine"].load(Path("r4"), 4)

    engine = holder["engine"] = _engine(glm, live=True, on_summary=reload)
    cfg = _cfg(max_compactions=1)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, cfg), allow_compaction=True)
    samples, events = _round(tmp_path, recorder, engine, Agent(4), n=1, live=True)
    trial = _by_trial(samples)[0]
    assert len(trial) == 2
    _check_segments(trial, engine, cfg, glm)
    versions = [(s["metadata"]["weight_version"], s["metadata"]["final_weight_version"]) for s in trial]
    assert versions == [("3", "3"), ("4", "4")]
    (group,) = [f for name, f in events if name == GROUP_EVENT]
    assert group["versions"] == [3, 4]  # the trial's oldest version is its first segment's


# ---------------------------------------------------------------------- finish_sample per segment
class _Segments:
    """A recorder whose ``close_segments`` returns fixed segments."""

    def __init__(self, parts: list[CompactionSegment]) -> None:
        self.parts = parts

    async def close(self, session):
        return [t for p in self.parts for t in p.turns]

    async def close_segments(self, session):
        return self.parts


def _turns(prompt: list[int], completions: list[list[int]]) -> tuple[TokenTurn, ...]:
    turns, stream = [], list(prompt)
    for completion in completions:
        turns.append(TokenTurn(prompt_ids=tuple(stream), completion_ids=tuple(completion),
                               completion_logprobs=(-0.5,) * len(completion), policy_version=3))
        stream += completion + [99]  # a tool output token
    return tuple(turns)


def _finish(parts, **kw):
    trajectory = _trajectory("t-0", TrialVerdict(rewards={"reward": 1.0}))
    return asyncio.run(finish_sample(trajectory, _Segments(parts), "t-0", HarnessMetrics(), segmented=True, **kw))


def test_finish_sample_joins_each_segment_and_refuses_a_cut_before_the_last(caplog):
    parts = [CompactionSegment(turns=_turns([1, 2], [[3, 4], [5]])),
             CompactionSegment(turns=_turns([1, 7, 8, 9], [[10, 11, 12]]), kept_steps=1)]
    done = _finish(parts, max_response_len=8)
    assert [p.tokens.prompt_ids for p in done.parts] == [(1, 2), (1, 7, 8, 9)]
    assert done.segments is done.parts[-1].tokens and done.trajectory.outcome is Outcome.OK
    with caplog.at_level(logging.WARNING, logger="yeto.rl.agentic.finish"):
        cut_last = _finish(parts, max_response_len=2)  # segment 0 has 4 response tokens: cut before the last
    assert cut_last.segments is None and cut_last.trajectory.reason.startswith("compaction_segment_truncated")
    assert "compaction segment 0" in caplog.text  # never a silent cut of a summary
    last_only = _finish([parts[1], parts[0]], max_response_len=3)  # 4 tokens in the last one only
    assert last_only.parts[-1].tokens.truncated and not last_only.parts[0].tokens.truncated
    # max_prompt_len bounds the first segment only: a later one starts from the summary and the last steps
    later = _finish(parts, max_response_len=8, max_prompt_len=3)  # segment 1's prompt has 4 tokens
    assert later.trajectory.outcome is Outcome.OK and len(later.parts) == 2
    first = _finish(parts, max_response_len=8, max_prompt_len=1)  # segment 0's prompt has 2
    assert first.segments is None and first.trajectory.reason == "prompt_exceeds_prompt_length"
    unsegmented = asyncio.run(finish_sample(_trajectory("t-0", TrialVerdict(rewards={"reward": 1.0})),
                                            _Segments(parts), "t-0", HarnessMetrics(), max_response_len=8))
    # without ``segmented`` (verl): one stream, as before; a compacted segment does not extend it
    assert unsegmented.parts == () and unsegmented.segments is None
    assert unsegmented.trajectory.reason.startswith("token_alignment")


def test_a_segment_without_a_trained_token_is_left_out_and_one_left_keeps_its_metadata():
    calls = _turns([1, 2], [[3, 4]])
    empty = (TokenTurn(prompt_ids=(1, 5), completion_ids=(), completion_logprobs=()),)
    done = _finish([CompactionSegment(turns=calls), CompactionSegment(turns=empty, kept_steps=0)],
                   max_response_len=8)
    assert len(done.parts) == 1 and done.turns == [*calls, *empty]  # every call still counts (versions)
    (sample,) = agentic_samples(ITEM, done, version=3, provenance={}, group_index=0, index=0)
    assert sample["metadata"]["compaction"]["num_segments"] == 1  # a compacted trial, its written segments
    nothing = _finish([CompactionSegment(turns=empty), CompactionSegment(turns=empty)], max_response_len=8)
    assert nothing.segments is None and nothing.trajectory.reason == "compaction_no_trained_tokens"


def test_joined_segments_keep_the_only_copy_of_their_rollout_arrays(glm):
    """A compacted trial holds one R3 array per segment (the joined one), not
    the calls' copies as well; the group drops them once written."""
    engine = _engine(glm, arrays=True)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, _cfg(), r3_response_len=4096), allow_compaction=True)
    done = _finished(recorder, 5, segmented=True)
    assert len(done.parts) == 4
    assert all(t.routed_experts is None and t.completion_top_ids is None for t in done.turns)
    assert all(p.tokens.routed_experts is not None and p.tokens.response_top_ids is not None for p in done.parts)


# ---------------------------------------------------------------------- verl still refuses
def test_the_verl_recorder_still_refuses_a_compacting_proxy(glm, monkeypatch):
    proxy = _proxy(_engine(glm), glm, _cfg())
    with pytest.raises(HarnessConfigError, match="CompactionRL is not supported"):
        ProxyTokenRecorder(proxy)
    module = type(sys)("t_cmp_records_proxy_factory")
    module.proxy = lambda: proxy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("YETO_RECORDS_PROXY", f"{module.__name__}:proxy")
    with pytest.raises(HarnessConfigError, match="CompactionRL is not supported"):
        recorder_from_env()  # verl's HarborAgentLoop and Miles' records generate build this one
    assert ProxyTokenRecorder(proxy, allow_compaction=True).compaction is proxy.backend.compaction


def test_the_engine_sampler_builds_its_recorder_with_compaction_allowed(monkeypatch, tmp_path):
    """``round_factory`` (the engine sampler's agentic island) is the one caller
    that reads the calls per segment, so it alone allows a compacting proxy."""
    from test_agentic_drain_defer import _built, _factory_world

    from yeto.rl.harness.gateway import recorder

    _factory_world(monkeypatch)
    built: list[dict] = []
    monkeypatch.setattr(recorder, "ProxyTokenRecorder", lambda *a, **k: built.append(k) or object())
    _built({"agentic": {}}, tmp_path)
    assert [k["allow_compaction"] for k in built] == [True]


def test_the_round_reads_turns_that_carry_no_policy_error():
    """Recorders and simulations may hand turns without ``policy_error`` (duck-typed)."""
    from types import SimpleNamespace

    from yeto.rl.engine.agentic_round import compaction_no_room

    assert compaction_no_room(SimpleNamespace(turns=[SimpleNamespace()])) is False
    assert compaction_no_room(SimpleNamespace(turns=[])) is False
