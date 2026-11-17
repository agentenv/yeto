"""Records proxy compaction: robustness of the summary turn (CPU).

A scripted :class:`FakeEngine` (``tests/test_records_proxy.py``) behind a
:class:`RecordsProxy` with a :class:`CompactionConfig`: the summary cut at its
length, the summary request that does not fit, a rebuilt context still over
the trigger, compactions used up, tool calls in the summary turn, configured
q_sum / u_resume, and the config checks for the formal 262144 / 10240 budget.
GLM-4.7-Flash keeps every turn's reasoning, so the summary request extends the
recorded stream (tokenizer from the local HF cache; skipped if absent).
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from test_records_proxy import (
    SCRIPT,
    SIGNED,
    ScriptedModel,
    _template,
    assistant,
    call,
    echo,
    engine_for,
    model_text,
    responses_body,
    run_codex,
    tool_output,
    turn_end,
    user_item,
)

from yeto.rl import compaction as cp
from yeto.rl.compaction import CompactionConfig, extract_summary, run_episode
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.records import OMITTED_FROM_SUMMARY, ContextFull


@pytest.fixture(scope="module")
def glm():
    return _template("glm5")


SUMMARY = assistant("check", content="<summary>1. listed files</summary>")


def run(proxy: Any, trajectory_id: str, results: list[str]) -> list[dict[str, Any]]:
    """``run_codex`` with the tool output of each turn given (``results[turn]``)."""
    items = [user_item("read a.txt")]
    outputs = []
    for turn, result in enumerate(results):
        response = asyncio.run(proxy.handle("responses", responses_body(copy.deepcopy(items)),
                                            trajectory_id=trajectory_id))
        outputs.append(response)
        items += echo(response["output"])
        for out in response["output"]:
            if out["type"] == "function_call":
                items.append(tool_output(out["call_id"], result))
    return outputs


def prompts(template: Any, replies: list[dict[str, Any]], results: list[str], **engine_kw: Any) -> list[int]:
    """Prompt lengths of each request of an uncompacted run (to place the trigger)."""
    dry = engine_for(ScriptedModel(template, replies), **engine_kw)
    run(RecordsProxy(dry, template, model="policy", signed_sampling=SIGNED), "dry", results)
    return [len(request[0][0]) for request in dry.token_requests]


def proxy_for(template: Any, replies: list[dict[str, Any]], cfg: CompactionConfig, **engine_kw: Any):
    engine = engine_for(ScriptedModel(template, replies), **engine_kw)
    return RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, compaction=cfg), engine


RESULTS = ["result 0", "result 1", "result 2"]


def third_trigger(template: Any, t_comp: int, **kw: Any) -> CompactionConfig:
    """Budget so the third request (after two tool steps) triggers, not the second."""
    third = prompts(template, list(SCRIPT), RESULTS)[2]
    return CompactionConfig(context_budget=third + t_comp - 1, t_comp=t_comp, **kw)


# ------------------------------------------------------------------ config
def test_formal_budget_defaults_and_summary_room_check():
    cfg = CompactionConfig(context_budget=262_144, t_comp=10_240)
    assert cfg.summary_prompt == cp.SUMMARY_PROMPT and cfg.resume_template == cp.RESUME_TEMPLATE
    assert cfg.summary_max_tokens is None and cfg.summary_tokens(256) == 256
    assert cfg.summary_tokens(32_768) == cp.DEFAULT_SUMMARY_MAX_TOKENS == 4096  # the turn cap, capped
    reserve = cp.summary_request_tokens(cp.SUMMARY_PROMPT)
    assert reserve == len(cp.SUMMARY_PROMPT.encode()) + cp.SUMMARY_REQUEST_OVERHEAD < 10_240
    assert cfg.check_summary_room(32_768) == 4096
    edge = CompactionConfig(context_budget=262_144, t_comp=10_240, summary_max_tokens=10_240 - reserve)
    assert edge.check_summary_room(65_536) == 10_240 - reserve  # the summary cap, not the turn cap
    small = CompactionConfig(context_budget=262_144, t_comp=4096)
    with pytest.raises(ValueError, match="cannot hold the summary turn"):
        small.check_summary_room(8192)  # 4096 + the request do not fit in t_comp 4096
    with pytest.raises(ValueError, match="summary_max_tokens <="):
        CompactionConfig(context_budget=262_144, t_comp=10_240, summary_max_tokens=10_240)
    with pytest.raises(ValueError, match="must be an integer"):
        CompactionConfig(context_budget=262_144, summary_max_tokens=True)
    with pytest.raises(ValueError, match=">= 1"):
        CompactionConfig(context_budget=262_144, summary_max_tokens=0)


def test_prompt_and_resume_template_overrides_are_checked():
    cfg = CompactionConfig(context_budget=1000, t_comp=10, summary_prompt="Summarize.",
                           resume_template="Earlier: {summary} {{literal}}")
    assert cfg.resume_template.format(summary="S") == "Earlier: S {literal}"
    for bad in ("no field", "{summary} {other}", "{summary} {0}", "{summary"):
        with pytest.raises(ValueError, match="resume_template"):
            CompactionConfig(context_budget=1000, t_comp=10, resume_template=bad)
    with pytest.raises(ValueError, match="summary_prompt"):
        CompactionConfig(context_budget=1000, t_comp=10, summary_prompt="  ")
    # a longer prompt needs more of t_comp
    with pytest.raises(ValueError, match="cannot hold"):
        CompactionConfig(context_budget=262_144, t_comp=10_240, summary_max_tokens=9000,
                         summary_prompt="x" * 2000)


def test_extract_summary_keeps_what_a_cut_reply_wrote():
    assert extract_summary("<analysis>a</analysis><summary> s </summary>") == ("s", True)
    assert extract_summary("<analysis>a</analysis><summary>1. goal\n2. done so") == ("1. goal\n2. done so", False)
    assert extract_summary("<analysis>going through the hist") == ("going through the hist", False)
    assert extract_summary("<analysis>a</analysis>plain") == ("plain", False)
    assert extract_summary("plain <analysis>cut") == ("plain", False)
    # a tag named inside the analysis is not the summary
    assert extract_summary("<analysis>then write <summary></analysis>body") == ("body", False)


def test_episode_driver_uses_the_configured_prompts():
    class Env:
        n = 0

        def step(self, action):
            self.n += 1
            return " ".join(["o"] * 30), self.n >= 4

        def reward(self):
            return 1.0

    seen = []

    def policy(messages):
        seen.append(messages[-1]["content"])
        return "<summary>S</summary>" if messages[-1]["content"] == "Q?" else "a " * 5

    cfg = CompactionConfig(context_budget=100, t_comp=40, summary_prompt="Q?", resume_template="R: {summary}")
    ep = run_episode(policy, Env(), [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}], cfg)
    assert ep.compactions >= 1 and "Q?" in seen
    assert ep.segments[1].prompt[1]["content"] == "R: S"


# ------------------------------------------------------------------ proxy
def test_summary_cut_at_its_length_still_opens_the_next_segment(glm):
    long = assistant("c", content="<summary>1. listed files " + "and more details " * 40 + "</summary>")
    cfg = third_trigger(glm, 2000, keep_recent_steps=1, summary_max_tokens=12)
    proxy, engine = proxy_for(glm, [SCRIPT[0], SCRIPT[1], long, SCRIPT[2]], cfg)
    outputs = run(proxy, "t", RESULTS)
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    assert engine.token_requests[2][1].max_tokens == 12  # the summary cap, not the turn's 256
    assert engine.token_requests[3][1].max_tokens == SIGNED["max_tokens"]
    records = proxy.records("t")
    first, second = records["sessions"][0]["samples"]
    assert first["turns"][-1]["kind"] == "summary" and first["turns"][-1]["finish_reason"] == "length"
    assert first["summary_ok"] is False and second["summary_ok"] is None
    assert records["counters"]["summary_truncated"] == 1 and records["counters"]["summary_not_ok"] == 1
    resume = glm.decode(second["token_ids"])
    assert "1. listed" in resume and "<summary>" not in resume  # the part written, tag dropped
    assert [t["kind"] for t in second["turns"]] == ["action"]
    assert second["turns"][0]["finish_reason"] == "stop" and second["turns"][0]["parse_error"] is None


def test_summary_request_without_room_leaves_the_newest_output_out(glm):
    """A tool output that fills the engine context: the summary turn would not
    fit (ContextFull before this change), so the summary request carries a note
    in its place; the rebuilt context cannot keep that step either (k = 0)."""
    results = ["result 0", "word " * 1500, "result 2"]
    third = prompts(glm, list(SCRIPT), results)[2]
    limit = third + 50  # the third prompt fits, the summary request on top of it does not
    cfg = CompactionConfig(context_budget=limit, t_comp=1000, keep_recent_steps=2, summary_max_tokens=64)
    proxy, engine = proxy_for(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]], cfg, max_model_len=limit)
    outputs = run(proxy, "t", results)
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t")
    assert records["counters"]["summary_newest_omitted"] == 1 and "context_full" not in records["counters"]
    first, second = records["sessions"][0]["samples"]
    assert [t["kind"] for t in first["turns"]] == ["action", "action", "summary"]
    assert first["summary_ok"] is True and second["kept_steps"] == 0
    request = glm.decode(first["token_ids"])
    assert OMITTED_FROM_SUMMARY in request and "word word" not in request
    assert len(engine.token_requests[2][0][0]) < third  # the summary prompt is the short one
    assert "1. listed files" in glm.decode(second["token_ids"])


def test_rebuilt_context_over_the_trigger_is_not_compacted_every_turn(glm):
    """A summary so long that even k = 0 crosses ``C - T_comp``: the next turns
    of that segment are not compacted again (max_compactions is not used up)."""
    replies = [SCRIPT[0], SCRIPT[1]]
    third = prompts(glm, [*replies, SCRIPT[1], SCRIPT[2]], ["result 0", "result 1", "result 2", "result 3"])[2]
    huge = assistant("c", content="<summary>" + "fact " * (third + 200) + "</summary>")
    cfg = CompactionConfig(context_budget=third + 2999, t_comp=3000, keep_recent_steps=1, max_compactions=3,
                           summary_max_tokens=2000)
    proxy, _ = proxy_for(glm, [*replies, huge, SCRIPT[1], SCRIPT[2]], cfg, max_model_len=8192)
    outputs = run(proxy, "t", ["result 0", "result 1", "result 2", "result 3"])
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t")
    assert records["counters"]["compactions"] == 1
    assert records["counters"]["compaction_rebuild_over_budget"] == 1
    first, second = records["sessions"][0]["samples"]
    assert second["kept_steps"] == 0 and [t["kind"] for t in second["turns"]] == ["action", "action"]
    assert all(t["parse_error"] is None for t in second["turns"])


def test_compactions_used_up_end_at_the_budget_as_context_full(glm):
    """After the last compaction the segment grows until the context reaches C
    (not the engine's larger max_model_len): ContextFull on its last turn; the
    segment before is a normal one ending with its summary."""
    cfg = third_trigger(glm, 600, keep_recent_steps=1, max_compactions=1)
    proxy, engine = proxy_for(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[1], SCRIPT[2]], cfg,
                              max_model_len=cfg.context_budget + 4096)
    with pytest.raises(ContextFull, match="context_budget"):
        run(proxy, "t", ["result 0", "result 1", "word " * 700, "result 3"])
    records = proxy.records("t")
    assert records["counters"]["compactions"] == 1
    assert records["counters"]["context_full"] == 1 and records["counters"]["context_budget_full"] == 1
    first, second = records["sessions"][0]["samples"]
    assert [t["kind"] for t in first["turns"]] == ["action", "action", "summary"]
    assert first["summary_ok"] is True and all(t["parse_error"] is None for t in first["turns"])
    assert first["turns"][-1]["finish_reason"] == "stop"
    assert [t["kind"] for t in second["turns"]] == ["action"]
    assert second["turns"][-1]["parse_error"].startswith("context_full: ")
    assert len(engine.token_requests) == 4  # the over-budget prompt never reached the engine


def test_tool_calls_in_the_summary_turn_are_not_run(glm):
    rogue = assistant("check", content="<summary>1. listed files</summary>", calls=[call("x", "rm -rf /tmp/w")])
    cfg = third_trigger(glm, 2000, keep_recent_steps=1)
    proxy, _ = proxy_for(glm, [SCRIPT[0], SCRIPT[1], rogue, SCRIPT[2]], cfg)
    outputs = run(proxy, "t", RESULTS)
    third = outputs[2]["output"]
    assert "function_call" not in [o["type"] for o in third]  # the agent got the next action, not the summary's call
    assert third[-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t")
    assert records["counters"]["summary_tool_calls"] == 1
    first, second = records["sessions"][0]["samples"]
    assert first["summary_ok"] is False
    resume = glm.decode(second["token_ids"])
    assert "1. listed files" in resume and "rm -rf" not in resume


def test_unparsable_summary_reply_uses_its_text(glm):
    class Raw(ScriptedModel):
        def __call__(self, prompt, index):
            if isinstance(self.replies[0], str):
                text = self.replies.pop(0)
                return self.template.encode(text) + self.template.encode("<|user|>")
            return super().__call__(prompt, index)

    bad = "<think>c</think><summary>1. listed files</summary><tool_call><arg_key>cmd</arg_key></tool_call>"
    cfg = third_trigger(glm, 2000, keep_recent_steps=1)
    engine = engine_for(Raw(glm, [SCRIPT[0], SCRIPT[1], bad, SCRIPT[2]]))  # type: ignore[list-item]
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED, compaction=cfg)
    outputs = run(proxy, "t", RESULTS)
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t")
    assert records["counters"]["summary_parse_errors"] == 1
    first, second = records["sessions"][0]["samples"]
    assert first["summary_ok"] is False and "1. listed files" in glm.decode(second["token_ids"])


def test_configured_summary_prompt_and_resume_template_are_used(glm):
    cfg = third_trigger(glm, 2000, keep_recent_steps=1, summary_prompt="Write <summary></summary> now.",
                        resume_template="Earlier work: {summary}")
    proxy, _ = proxy_for(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]], cfg)
    run(proxy, "t", RESULTS)
    first, second = proxy.records("t")["sessions"][0]["samples"]
    assert "Write <summary></summary> now." in glm.decode(first["token_ids"])
    assert cp.SUMMARY_PROMPT not in glm.decode(first["token_ids"])
    resume = glm.decode(second["token_ids"])
    assert "Earlier work: 1. listed files" in resume and "continues from an earlier one" not in resume


def test_a_budget_never_reached_records_what_no_compaction_records(glm):
    plain_engine = engine_for(ScriptedModel(glm, list(SCRIPT)))
    plain = RecordsProxy(plain_engine, glm, model="policy", signed_sampling=SIGNED)
    run_codex(plain, "t")
    cfg = CompactionConfig(context_budget=3000, t_comp=1000, summary_max_tokens=128)
    proxy, engine = proxy_for(glm, list(SCRIPT), cfg)
    run_codex(proxy, "t")
    records, expected = proxy.records("t"), plain.records("t")
    # the gateway counts compaction forks / replays only for a compacting backend (C8)
    extra = {k: records["gateway"].pop(k) for k in set(records.get("gateway") or {}) - set(expected.get("gateway") or {})}
    assert set(extra) <= {"retry_replays", "compaction_fork_lost"} and not any(extra.values())
    assert records == expected
    assert [r[1] for r in engine.token_requests] == [r[1] for r in plain_engine.token_requests]


# ------------------------------------------------------------------ the engine context (C5 P5 / P16)
def test_summary_headroom_check_for_the_formal_budget():
    cfg = CompactionConfig(context_budget=262_144, t_comp=10_240)
    request = cp.summary_request_tokens(cp.SUMMARY_PROMPT)
    over = 262_144 - 10_240 + 8192 + request + 4096 - 262_144
    with pytest.raises(ValueError, match=f"overrun max_model_len 262144 by {over} tokens"):
        cfg.check_summary_headroom(262_144, 8192)  # C = max_model_len, 8192 turns, a 4096 summary
    fits = CompactionConfig(context_budget=262_144 - over, t_comp=10_240)
    assert fits.check_summary_headroom(262_144, 8192) == 0
    capped = CompactionConfig(context_budget=262_144 - 16_384, t_comp=10_240, summary_max_tokens=4096)
    assert capped.check_summary_headroom(262_144, 8192) == 16_384 - (8192 + request + 4096 - 10_240)


def _stream_after_first_action(template: Any, results: list[str]) -> int:
    dry = engine_for(ScriptedModel(template, list(SCRIPT)))
    proxy = RecordsProxy(dry, template, model="policy", signed_sampling=SIGNED)
    run(proxy, "dry", results)
    first = proxy.records("dry")["sessions"][0]["samples"][0]["turns"][0]
    return len(first["prompt_added"]) + len(first["completion"])


def test_summary_request_that_cannot_fit_ends_as_context_full_before_the_engine(glm):
    """Even without the newest output the summary request is over max_model_len
    (a budget the headroom check refuses): the trajectory ends as context full,
    marked on the last action, and nothing is sent to the engine."""
    results = ["word " * 600, "result 1", "result 2"]
    limit = _stream_after_first_action(glm, results) + 20
    cfg = CompactionConfig(context_budget=limit, t_comp=10, keep_recent_steps=1)
    proxy, engine = proxy_for(glm, list(SCRIPT), cfg, max_model_len=limit)
    with pytest.raises(ContextFull, match="compaction_no_room: the summary request"):
        run(proxy, "t", results)
    records = proxy.records("t")
    assert records["counters"]["compaction_no_room"] == 1 and records["counters"]["context_full"] == 1
    assert records["counters"]["summary_newest_omitted"] == 1
    (only,) = records["sessions"][0]["samples"]
    assert [t["kind"] for t in only["turns"]] == ["action"]
    assert only["turns"][-1]["parse_error"].startswith("context_full: compaction_no_room: the summary request")
    assert len(engine.token_requests) == 1


def test_rebuilt_context_over_max_model_len_ends_after_the_summary(glm):
    cfg = third_trigger(glm, 2000, keep_recent_steps=1, resume_template="{summary}" + " pad" * 5000)
    proxy, engine = proxy_for(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]], cfg)
    with pytest.raises(ContextFull, match="compaction_no_room: the rebuilt context"):
        run(proxy, "t", RESULTS)
    records = proxy.records("t")
    assert records["counters"]["compaction_no_room"] == 1
    (only,) = records["sessions"][0]["samples"]  # no empty segment was opened
    assert [t["kind"] for t in only["turns"]] == ["action", "action", "summary"]
    assert only["turns"][-1]["parse_error"].startswith("context_full: compaction_no_room: the rebuilt context")
    assert len(engine.token_requests) == 3


def test_turn_cut_at_max_model_len_is_context_full_with_compaction_only(glm):
    """A reply the engine cut because the stream reached max_model_len, continued
    in the same stream: with compaction the next request ends the trajectory as
    context full (it was a RecordsMismatch, an infrastructure abort); without
    compaction it stays one."""
    from yeto.rl.harness.gateway.records import RecordingBackend, RecordsMismatch

    history = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "read a.txt"}]
    first = len(glm.encode(glm.render(history, add_generation_prompt=True)))
    limit = first + 3  # the reply is cut after 3 tokens, at the end of the engine context
    for compaction, error in ((CompactionConfig(context_budget=limit, t_comp=1), ContextFull), (None, RecordsMismatch)):
        engine = engine_for(ScriptedModel(glm, [SCRIPT[2]]), max_model_len=limit)
        backend = RecordingBackend(engine, glm, compaction=compaction)

        async def go():
            session = await backend.create_session("t")
            reply = await backend.chat(session, list(history), dict(SIGNED))
            assert reply["choices"][0]["finish_reason"] == "length"
            await backend.chat(session, [*history, reply["choices"][0]["message"],
                                         {"role": "user", "content": "go on"}], dict(SIGNED))

        with pytest.raises(error):
            asyncio.run(go())
        (only,) = backend.records("t")["sessions"][0]["samples"]
        assert only["turns"][-1]["finish_reason"] == "length" and len(only["token_ids"]) == limit
        parse_error = only["turns"][-1]["parse_error"]
        if compaction is None:
            assert parse_error is None
        else:
            assert parse_error.startswith("context_full: the stream has")


def test_a_trajectory_forgotten_during_its_summary_turn_samples_nothing_more(glm):
    """The trial was cut (drain / detach) while the engine wrote the summary: the
    request ends with RecordsMismatch and no next segment is sampled."""
    cfg = third_trigger(glm, 2000, keep_recent_steps=1)
    script = ScriptedModel(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]])
    engine = engine_for(script)
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED, compaction=cfg)

    def reply(prompt, index):
        if len(script.replies) == 2:  # the summary request
            proxy.backend.forget("t")
        return script(prompt, index)

    engine.token_reply = reply
    from yeto.rl.harness.gateway.records import RecordsMismatch

    with pytest.raises(RecordsMismatch, match="forgotten while a request was in flight"):
        run(proxy, "t", RESULTS)
    assert len(engine.token_requests) == 3 and script.replies == [SCRIPT[2]]
    assert proxy.backend.sessions == {} and proxy.backend.counters == {}


# C4's ``tcomp_overflow`` scenario (scripts/sim_compaction_grpo.py on cmp1118/C4, parameters only):
# C = max_model_len = 4096, t_comp 512, k 2, 3 compactions, 256-token turns, 9-12 steps of a
# 30-90 token action and a 300-450 token tool output, 60-120 token summaries. Each step is longer
# than t_comp leaves, so the step that triggers a compaction has already passed C - t_comp by a
# whole step and the summary request on top of it overflows max_model_len (11 of 35 compactions
# died there before the summary request left the newest output out).
TCOMP_OVERFLOW = {"context_budget": 4096, "t_comp": 512, "keep_recent_steps": 2, "max_compactions": 3}
_WORDS = ("alpha", "beta", "gamma", "delta", "file", "line", "value", "error", "test", "build", "path", "node")


def _words(rng: Any, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


@pytest.mark.parametrize("seed", range(6))
def test_c4_tcomp_overflow_scenario_compacts_without_running_out_of_room(glm, seed):
    import random

    rng = random.Random(seed)
    steps = rng.randint(9, 12)
    state = {"actions": 0, "summaries": 0}
    def reply(prompt, index):
        if cp.SUMMARY_PROMPT[-40:] in glm.decode(list(prompt[-300:])):  # the summary request
            state["summaries"] += 1
            message = assistant(_words(rng, 10), content=f"<analysis>{_words(rng, rng.randint(20, 40))}</analysis>"
                                f"<summary>1. step {state['actions']} {_words(rng, rng.randint(60, 120))}</summary>")
        elif state["actions"] < steps:
            state["actions"] += 1
            message = assistant(_words(rng, rng.randint(30, 90)), calls=[call("x", f"run {state['actions']}")])
        else:
            message = assistant("done", content="The task is done.")
        return glm.encode(model_text(glm, message)) + glm.encode(turn_end(glm, message))[:1]

    engine = engine_for(ScriptedModel(glm, []), max_model_len=TCOMP_OVERFLOW["context_budget"])
    engine.token_reply = reply
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED,
                         compaction=CompactionConfig(**TCOMP_OVERFLOW))
    items, finished = [user_item("fix the build")], False
    for _ in range(steps + 1):
        response = asyncio.run(proxy.handle("responses", responses_body(copy.deepcopy(items)), trajectory_id="t"))
        items += echo(response["output"])
        calls = [o for o in response["output"] if o["type"] == "function_call"]
        if not calls:
            finished = response["output"][-1]["content"][0]["text"] == "The task is done."
            break
        items += [tool_output(o["call_id"], _words(rng, rng.randint(300, 450))) for o in calls]
    records = proxy.records("t")
    counters = records["counters"]
    assert finished, counters
    assert "context_full" not in counters and "compaction_no_room" not in counters
    samples = [s for session in records["sessions"] for s in session["samples"]]
    assert counters.get("compactions", 0) == state["summaries"] == len(samples) - 1 >= 1
    for sample in samples[:-1]:  # every summary was written in full and opened the next segment
        assert sample["summary_ok"] is True and sample["turns"][-1]["finish_reason"] == "stop"
    assert all(s["kept_steps"] >= 1 for s in samples[1:])  # the omitted output is kept verbatim


def test_a_summary_drafted_in_the_reasoning_is_not_the_summary(glm):
    """C5 P12: the summary comes from the response channel; a ``<summary>`` the
    model drafts while thinking is neither taken nor counted as ok."""
    drafted = assistant("plan: <summary></summary> then more", content="<summary>1. listed files</summary>")
    cfg = third_trigger(glm, 2000, keep_recent_steps=1)
    proxy, _ = proxy_for(glm, [SCRIPT[0], SCRIPT[1], drafted, SCRIPT[2]], cfg)
    run(proxy, "t", RESULTS)
    first, second = proxy.records("t")["sessions"][0]["samples"]
    assert first["summary_ok"] is True and "1. listed files" in glm.decode(second["token_ids"])

    thinking = assistant("<summary></summary> " + "keep thinking " * 40, content="<summary>never written</summary>")
    cfg = third_trigger(glm, 2000, keep_recent_steps=1, summary_max_tokens=12)  # cut inside the reasoning
    proxy, _ = proxy_for(glm, [SCRIPT[0], SCRIPT[1], thinking, SCRIPT[2]], cfg)
    run(proxy, "t", RESULTS)
    first, second = proxy.records("t")["sessions"][0]["samples"]
    assert first["summary_ok"] is False and first["turns"][-1]["finish_reason"] == "length"
    assert "never written" not in glm.decode(second["token_ids"])


# ------------------------------------------------------------------ R1: forget races, drift counts
def test_a_trajectory_forgotten_during_an_action_turn_leaves_no_counters(glm):
    """forget() while the engine samples a turn that appends a stop token the
    engine did not return (the stop_added counter): RecordsMismatch, no KeyError,
    and no counter entry is made again for the forgotten trajectory."""
    from yeto.rl.harness.gateway.records import RecordsMismatch

    script = ScriptedModel(glm, [SCRIPT[0], SCRIPT[1], SCRIPT[2]], return_stop=False)
    engine = engine_for(script)
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED)

    def reply(prompt, index):
        if len(script.replies) == 2:  # the second turn: its prompt starts with the stop the engine did not return
            proxy.backend.forget("t")
        return script(prompt, index)

    engine.token_reply = reply
    with pytest.raises(RecordsMismatch, match="forgotten while a request was in flight"):
        run(proxy, "t", RESULTS)
    assert proxy.backend.sessions == {} and proxy.backend.counters == {} and len(engine.token_requests) == 2


def test_a_request_for_a_forgotten_session_is_a_records_mismatch(glm):
    from yeto.rl.harness.gateway.records import RecordingBackend, RecordsMismatch

    backend = RecordingBackend(engine_for(ScriptedModel(glm, [SCRIPT[0]])), glm)

    async def go():
        session = await backend.create_session("t")
        backend.forget("t")
        await backend.chat(session, [{"role": "user", "content": "x"}], dict(SIGNED))

    with pytest.raises(RecordsMismatch, match="forgotten"):
        asyncio.run(go())
    assert backend.counters == {}


def test_drift_is_counted_once_per_sampled_turn_on_a_compaction_turn(glm):
    """The second reply is sampled in a non-canonical tokenization, so every later
    prompt of that segment drifts. The compacting request builds two prompts
    (the agent's turn, then the summary request) and samples one: the counter
    equals the sampled turns that carry a render_diff."""

    class Spelled(ScriptedModel):
        def __call__(self, prompt, index):
            ids = super().__call__(prompt, index)
            if len(self.replies) == 2:  # the second reply
                text = self.template.decode(ids[:3])
                ids = [i for ch in text for i in self.template.encode(ch)] + ids[3:]
            return ids

    cfg = third_trigger(glm, 2000, keep_recent_steps=1)
    engine = engine_for(Spelled(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]]))
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED, compaction=cfg)
    run(proxy, "t", RESULTS)
    records = proxy.records("t")
    assert records["counters"]["compactions"] == 1
    turns = [t for s in records["sessions"][0]["samples"] for t in s["turns"]]
    assert turns[2]["kind"] == "summary" and turns[2]["render_diff"] is not None  # the summary request drifted
    assert records["counters"]["render_drift_turns"] == sum(t["render_diff"] is not None for t in turns) == 1
