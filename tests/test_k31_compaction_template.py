"""Records proxy compaction on the real chat templates (CPU).

A pi-like agent loop (chat completions, the full history every turn, long tool
outputs) drives :class:`RecordsProxy` with compaction over a scripted
:class:`FakeEngine`: each reply is the template's own rendering of the next
scripted assistant message, so a correct proxy shows no render drift. Checked
per segment: the engine prompts are prefixes of the segment's stream, the
stream is the template rendering of the segment's context token for token,
and a later segment starts at ``system + user(resume) + the last k steps``.

Kimi-K3.1 needs the checkpoint's tokenizer (``YETO_K31_TOKENIZER``, as
``tests/test_kimi_k31_template.py``; skipped without it); Qwen3-0.6B comes from
the HF cache (``tests/hf_assets.py``).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
from typing import Any

import pytest
from hf_assets import require_hf_repo

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.compaction import RESUME_TEMPLATE, SUMMARY_PROMPT, CompactionConfig
from yeto.rl.harness.chat_templates import TEMPLATES, ChatTemplate, check_alignment, parse_kimi_k3
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.records import ContextFull, RecordsMismatch

SIGNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}, "timeout": {"type": "number"}},
                                               "required": ["command"]}}}]
SYSTEM = "You are pi, a coding agent.\nUse tools."
TASK = "Find the failing test in /app and fix it."
STEPS = 14  # tool steps before the final answer

K31_TOKENIZER = os.environ.get("YETO_K31_TOKENIZER", "")
needs_k31 = pytest.mark.skipif(not K31_TOKENIZER or not os.path.isfile(os.path.join(K31_TOKENIZER, "tiktoken.model")),
                               reason="set YETO_K31_TOKENIZER to the K3.1 tokenizer files")


def _with_kwargs(template: ChatTemplate, kwargs: dict[str, Any]) -> ChatTemplate:
    """As ``agentic_round`` applies ``agentic.template_kwargs`` (merged over the spec's)."""
    spec = dataclasses.replace(template.spec, template_kwargs={**template.spec.template_kwargs, **kwargs})
    return ChatTemplate(spec, template.tokenizer)


@pytest.fixture(scope="module")
def k31_base() -> ChatTemplate:
    return ChatTemplate.load("kimi_k31", tokenizer_path=K31_TOKENIZER, trust_remote_code=True)


@pytest.fixture(scope="module")
def qwen3_base() -> ChatTemplate:
    return ChatTemplate.load("qwen3", tokenizer_path=require_hf_repo(TEMPLATES["qwen3"].tokenizer))


# ---------------------------------------------------------------------- scripted model
def _action(step: int) -> dict[str, Any]:
    args = json.dumps({"command": f"cat /app/tests/test_{step}.py | head -n 40", "timeout": 30})
    message = {"role": "assistant", "content": "" if step % 2 else f"Looking at test {step}.",
               "reasoning_content": f"Step {step}: the previous output says the fixture is stale; read the next file.",
               "tool_calls": [{"id": f"c{step}", "type": "function", "function": {"name": "bash", "arguments": args}}]}
    return message


SUMMARY = {"role": "assistant", "reasoning_content": "Collect what must be kept before the context is cut.",
           "content": "<analysis>steps in order</analysis>\n<summary>1. Goal: fix the failing test in /app\n"
                      "2. Completed actions: read the test files\n3. Next: patch the fixture</summary>"}
FINAL = {"role": "assistant", "reasoning_content": "All files read.", "content": "Fixed the stale fixture."}


def _thinking(template: ChatTemplate) -> bool:
    kwargs = template.spec.template_kwargs
    return bool(kwargs.get("thinking", kwargs.get("enable_thinking", True)))


def _as_generated(template: ChatTemplate, message: dict[str, Any]) -> dict[str, Any]:
    """The message as the template's parser gives it back (a non-thinking turn has no reasoning)."""
    return message if _thinking(template) else {k: v for k, v in message.items() if k != "reasoning_content"}


def reply_ids(template: ChatTemplate, message: dict[str, Any], *, return_stop: bool) -> list[int]:
    """Ids the model samples for ``message`` after the generation prompt, through its stop token."""
    user = [{"role": "user", "content": "x"}]
    opener = template.render(user, add_generation_prompt=True)
    opener_ids = template.encode(opener)
    full = template.render(user + [_as_generated(template, message)], add_generation_prompt=False)
    assert full.startswith(opener), "the generation prompt is not a prefix of the turn's rendering"
    ids = template.encode(full)
    assert ids[:len(opener_ids)] == opener_ids
    body = ids[len(opener_ids):]
    stops = [i for i, t in enumerate(body) if t in template.stop_token_ids]
    if stops:  # the template closes the turn with a stop token (Kimi, Qwen)
        body = body[:stops[0] + 1]
    else:  # GLM: the turn ends where the model opens the next role (tool output or user)
        end = "<|observation|>" if message.get("tool_calls") else "<|user|>"
        body = template.encode(template.decode(body).rstrip("\n")) + template.encode(end)
    return body if return_stop else body[:-1]


class Model:
    """``token_reply``: the summary when the prompt ends in the summary request, else the next action."""

    def __init__(self, template: ChatTemplate, *, return_stop: bool = True) -> None:
        self.template = template
        self.return_stop = return_stop
        self.actions = [_action(i) for i in range(STEPS)] + [FINAL]
        self.summaries = 0
        tail = template.render([{"role": "user", "content": SUMMARY_PROMPT}], add_generation_prompt=True)
        self.summary_tail = tail[tail.rfind(SUMMARY_PROMPT):]

    def __call__(self, prompt: tuple[int, ...], index: int) -> list[int]:
        if self.template.decode(prompt).endswith(self.summary_tail):
            self.summaries += 1
            message = SUMMARY
        else:
            message = self.actions.pop(0)
        return reply_ids(self.template, message, return_stop=self.return_stop)


def tool_output(step: int) -> str:
    return "\n".join(f"line {i} of test_{step}.py: assert fixture_{i % 7}() == {i * step}" for i in range(24))


# ---------------------------------------------------------------------- agent loop
@dataclasses.dataclass
class Run:
    proxy: RecordsProxy
    engine: FakeEngine
    messages: list[dict[str, Any]]  # the agent's final history
    triggers: list[list[dict[str, Any]]]  # per compaction: the agent request that triggered it
    model: Model


def run_agent(template: ChatTemplate, cfg: CompactionConfig | None, *, return_stop: bool = True) -> Run:
    model = Model(template, return_stop=return_stop)
    engine = FakeEngine(token_reply=model, max_model_len=65536)
    engine.start(None, 3)
    proxy = RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, compaction=cfg, max_chains=1)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    triggers = []
    for turn in range(STEPS + 1):
        before = proxy.backend.counters.get("t", {}).get("compactions", 0)
        try:
            response = asyncio.run(proxy.handle("chat", {"model": "policy", "messages": copy.deepcopy(messages),
                                                         "tools": TOOLS}, trajectory_id="t"))
        except ContextFull:
            # the compactions are used up and the last segment reached context_budget: the
            # trajectory ends (truncated) at its last sampled turn; the unanswered tool outputs
            # never entered the stream
            assert cfg is not None and len(triggers) == cfg.max_compactions
            assert proxy.backend.counters["t"]["context_budget_full"] == 1
            while messages[-1]["role"] != "assistant":
                messages.pop()
            return Run(proxy, engine, messages, triggers, model)
        if proxy.backend.counters["t"].get("compactions", 0) > before:
            triggers.append(copy.deepcopy(messages))
        message = response["choices"][0]["message"]
        messages.append(message)  # pi sends the assistant message back as it came (reasoning included)
        for call in message.get("tool_calls") or []:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_output(turn)})
    assert messages[-1]["content"] == FINAL["content"]
    return Run(proxy, engine, messages, triggers, model)


def _step_tokens(template: ChatTemplate) -> tuple[int, int]:
    """(tokens of system + task + generation prompt, tokens one tool step adds)."""
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    one = base + [_as_generated(template, _action(0)),
                  {"role": "tool", "tool_call_id": "c0", "content": tool_output(0)}]
    n0 = len(template.encode(template.render(base, tools=TOOLS, add_generation_prompt=True)))
    n1 = len(template.encode(template.render(one, tools=TOOLS, add_generation_prompt=True)))
    return n0, n1 - n0


def _config(template: ChatTemplate, *, keep: int, max_compactions: int) -> CompactionConfig:
    """A budget that triggers after about four steps (the summary request still fits)."""
    n0, step = _step_tokens(template)
    return CompactionConfig(context_budget=n0 + 4 * step + 50, t_comp=50, keep_recent_steps=keep,
                            max_compactions=max_compactions)


def check_segments(template: ChatTemplate, run: Run, *, prefill: str = "") -> list[dict[str, Any]]:
    """Every segment joins (engine prompts are prefixes of its stream, the mask marks
    exactly the sampled ids) and equals the rendering of its context token for token
    (``prefill``: generation-prompt text the history form omits, counted as drift)."""
    records = run.proxy.records("t")
    (session,) = records["sessions"]
    samples = session["samples"]
    backend_segments = run.proxy.backend.sessions[session["session_id"]].segments
    drift = records["counters"].get("render_drift_turns", 0)
    assert drift == (records["counters"].get("prefill_dropped_turns", 0) if prefill else 0), records["counters"]
    assert records["counters"].get("compactions", 0) == len(samples) - 1 == len(run.triggers)
    prompts = iter(p[0] for p, *_ in run.engine.token_requests)
    for index, (sample, segment) in enumerate(zip(samples, backend_segments)):
        ids, mask = sample["token_ids"], sample["loss_mask"]
        joined, sampled = [], []
        for turn in sample["turns"]:  # a turn's stop_added: the stop token appended before the next turn
            assert list(next(prompts)) == [*joined, *turn["prompt_added"]]
            joined += [*turn["prompt_added"], *turn["completion"], *turn["stop_added"]]
            sampled += [0] * len(turn["prompt_added"]) + [1] * len(turn["completion"]) + [0] * len(turn["stop_added"])
        assert joined == ids and sampled == mask
        last = index == len(samples) - 1
        kinds = [t["kind"] for t in sample["turns"]]
        assert kinds[-1] == ("action" if last else "summary") and kinds.count("summary") == (0 if last else 1)
        # the context this segment's stream renders
        if last:
            context = segment.head + run.messages[segment.base:]
        else:
            trigger = run.triggers[index]
            context = segment.head + trigger[segment.base:] + [{"role": "user", "content": SUMMARY_PROMPT},
                                                               _as_generated(template, SUMMARY)]
        rendered = template.render(context, tools=TOOLS, add_generation_prompt=False)
        if prefill:  # Qwen3 without thinking: the stream keeps each generation prompt's empty think prefill
            stream = template.decode(ids).replace(prefill, "")
            assert stream == rendered.replace(prefill, "")[:len(stream)]
        else:
            expected = template.encode(rendered)
            # GLM's stop opens the next role: the rendering of the last turn ends before it
            body = ids[:-1] if len(ids) > len(expected) and ids[-1] in template.stop_token_ids else ids
            assert body == expected[:len(body)], f"segment {index}: differs at {first_diff(body, expected)}"
            rest = template.decode([t for t in expected[len(body):] if t not in template.stop_token_ids])
            assert rest.strip() == "", f"segment {index}: the rendering goes on after the stream: {rest[:80]!r}"
        assert check_alignment(template, context, tools=TOOLS, add_generation_prompt=False).aligned
        if index:  # system + user(resume) + the last k steps, each step starting at an assistant message
            assert [m["role"] for m in segment.head] == ["system", "user"]
            assert segment.head[1]["content"].startswith(RESUME_TEMPLATE.split("{summary}")[0])
            assert "1. Goal: fix the failing test" in segment.head[1]["content"]
            assert run.messages[segment.base]["role"] == "assistant"
            kept = run.triggers[index - 1][segment.base:]  # what the compaction carried over
            assert sum(m["role"] == "assistant" for m in kept) == sample["kept_steps"]
    return samples


def first_diff(a: list[int], b: list[int]) -> int | None:
    end = None if len(a) == len(b) else min(len(a), len(b))
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), end)


# ---------------------------------------------------------------------- Kimi-K3.1
K31_KWARGS = [{"thinking_effort": "low"}, {}, {"thinking_effort": "max"}, {"thinking": False}]


@needs_k31
@pytest.mark.parametrize("kwargs", K31_KWARGS, ids=["low", "default", "max", "no_thinking"])
@pytest.mark.parametrize("max_compactions", [1, 3])
def test_k31_compaction_segments_join_without_drift(k31_base, kwargs, max_compactions):
    template = _with_kwargs(k31_base, kwargs)
    run = run_agent(template, _config(template, keep=2, max_compactions=max_compactions))
    samples = check_segments(template, run)
    assert len(samples) == 1 + max_compactions
    assert run.proxy.records("t")["counters"].get("prefill_dropped_turns", 0) == 0
    if kwargs.get("thinking", True):  # every turn's reasoning stays in the stream, kept steps included
        text = template.decode(samples[-1]["token_ids"])
        actions = run.messages[run.proxy.backend.sessions["rec0"].segments[-1].base:]
        assert text.count("the fixture is stale") == sum(m["role"] == "assistant" and "tool_calls" in m for m in actions)


@needs_k31
def test_k31_compaction_when_the_engine_does_not_return_the_stop_token(k31_base):
    template = _with_kwargs(k31_base, {"thinking_effort": "low"})
    run = run_agent(template, _config(template, keep=2, max_compactions=2), return_stop=False)
    check_segments(template, run)
    assert run.proxy.records("t")["counters"]["stop_added"] > 0


@needs_k31
@pytest.mark.parametrize("keep", [0, 1])
def test_k31_compaction_keeping_fewer_steps(k31_base, keep):
    """k = 0: the next segment starts at system + user(resume) and the next agent turn's messages."""
    template = _with_kwargs(k31_base, {"thinking_effort": "low"})
    run = run_agent(template, _config(template, keep=keep, max_compactions=2))
    records = run.proxy.records("t")
    assert records["counters"].get("render_drift_turns", 0) == 0
    samples = records["sessions"][0]["samples"]
    assert len(samples) == 3 and all(s["kept_steps"] == keep for s in samples[1:])
    check_segments(template, run)


@needs_k31
def test_k31_generation_prompt_prefixes_the_rebuilt_context(k31_base):
    """The rebuilt context (system + user(resume) + steps starting at an assistant
    message): its generation prompt plus the sampled text is a prefix of the next
    rendering, tool outputs resolve their tool by the kept assistant message."""
    template = _with_kwargs(k31_base, {"thinking_effort": "low"})
    rebuilt = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": RESUME_TEMPLATE.format(summary="1. Goal: x")},
               _action(3), {"role": "tool", "tool_call_id": "c3", "content": tool_output(3)}]
    prompt = template.render(rebuilt, tools=TOOLS, add_generation_prompt=True)
    message = parse_kimi_k3(template.decode(reply_ids(template, _action(4), return_stop=False)))
    text = template.decode(reply_ids(template, _action(4), return_stop=True))
    after = template.render(rebuilt + [message], tools=TOOLS, add_generation_prompt=False)
    assert after.startswith(prompt + text)
    assert '<|open|>message role="tool" tool="bash" index="1"<|sep|>' in prompt
    assert check_alignment(template, rebuilt, tools=TOOLS).aligned


# ---------------------------------------------------------------------- Qwen3
def test_qwen3_without_thinking_compacts(qwen3_base):
    """The smoke-test setting for Qwen3-0.6B (``agentic.template_kwargs:
    {"enable_thinking": false}``): no reasoning in any turn, so the summary
    request does not rewrite history; only the generation prompt's empty think
    prefill is dropped from the history form (template text, mask 0)."""
    template = _with_kwargs(qwen3_base, {"enable_thinking": False})
    run = run_agent(template, _config(template, keep=2, max_compactions=2))
    samples = check_segments(template, run, prefill="<think>\n\n</think>\n\n")
    assert len(samples) == 3
    assert run.proxy.records("t")["counters"]["prefill_dropped_turns"] > 0


def test_qwen3_with_thinking_fails_closed_at_the_summary_request(qwen3_base):
    """Qwen3 thinking drops the reasoning of every turn before the last user
    message: the summary request (a user message) rewrites the segment's
    history, so the proxy refuses instead of training a stream it did not sample."""
    template = _with_kwargs(qwen3_base, {"enable_thinking": True})
    with pytest.raises(RecordsMismatch, match="rendering of the earlier turns changed"):
        run_agent(template, _config(template, keep=2, max_compactions=1))
    # without compaction the same loop is one aligned stream (no user message after the first)
    run = run_agent(template, None)
    (sample,) = run.proxy.records("t")["sessions"][0]["samples"]
    assert run.proxy.records("t")["counters"].get("render_drift_turns", 0) == 0
    expected = template.encode(template.render(run.messages, tools=TOOLS, add_generation_prompt=False))
    assert sample["token_ids"] == expected[:len(sample["token_ids"])]
    assert template.decode(expected[len(sample["token_ids"]):]).strip() == ""


@needs_k31
def test_k31_summary_comes_from_the_response_channel(k31_base, monkeypatch):
    """The summary request asks for ``<summary></summary>``; a thinking model often
    quotes the tags in its reasoning, and without tags the fallback is the whole
    completion. The resume message should carry the response channel's summary only."""
    template = _with_kwargs(k31_base, {"thinking_effort": "low"})
    quoting = dict(SUMMARY, reasoning_content="I must write <summary></summary> with the numbered sections.")
    untagged = dict(SUMMARY, content="Goal: fix the failing test in /app.")
    resumes = []
    for summary in (quoting, untagged):
        monkeypatch.setattr(__import__(__name__), "SUMMARY", summary)
        run = run_agent(template, _config(template, keep=2, max_compactions=1))
        resumes.append(run.proxy.backend.sessions["rec0"].segments[1].head[1]["content"])
    assert "1. Goal: fix the failing test" in resumes[0]
    assert "Goal: fix the failing test in /app." in resumes[1]
    assert not any("<|close|>" in r or "numbered sections" in r for r in resumes)


# ---------------------------------------------------------------------- the other thinking templates
@pytest.mark.parametrize("name", ["glm5", "qwen38"])
@pytest.mark.parametrize("thinking", [True, False])
def test_thinking_templates_compact_with_and_without_thinking(name, thinking):
    """GLM (clear_thinking=False) and Qwen3.8 (preserve_thinking) keep every turn's
    reasoning, so the summary request appends; with ``enable_thinking: false``
    the parser reads the output after the closed reasoning block."""
    base = ChatTemplate.load(name, tokenizer_path=require_hf_repo(TEMPLATES[name].tokenizer))
    template = _with_kwargs(base, {"enable_thinking": thinking})
    run = run_agent(template, _config(template, keep=2, max_compactions=3))
    counters = run.proxy.records("t")["counters"]
    prefill = "<think>\n\n</think>\n\n" if counters.get("prefill_dropped_turns") else ""
    samples = check_segments(template, run, prefill=prefill)
    assert len(samples) == 4


def test_parsers_follow_the_thinking_switch():
    """With thinking off the generation prompt closed the reasoning block: the output
    is content and tool calls (an open-block parser read it all as reasoning)."""
    from yeto.rl.harness.chat_templates import parse_glm, parse_qwen38

    glm = "ok<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
    qwen38 = "ok\n\n<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>"
    for parse, text in ((parse_glm, glm), (parse_qwen38, qwen38)):
        message = parse(text, thinking=False)
        assert message["content"] == "ok" and "reasoning_content" not in message
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        assert parse(text)["reasoning_content"] == text.strip("\n")  # thinking on: still inside the block
    assert {n: s.thinking_kwarg for n, s in TEMPLATES.items()} == {
        "glm5": "enable_thinking", "qwen3": None, "qwen3_2507": None, "qwen38": "enable_thinking",
        "kimi_k31": "thinking", "nemotron3": "enable_thinking"}


@needs_k31
def test_k31_thinking_effort_is_fixed_per_stream_and_tool_choice_is_not(k31_base):
    """The thinking-effort message renders before the first message, so a summary
    turn with another effort would rewrite the stream's head (not an append);
    ``tool_choice`` renders after the last message, so it changes only that prompt."""
    conversation = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    low = _with_kwargs(k31_base, {"thinking_effort": "low"}).render(conversation, tools=TOOLS, add_generation_prompt=True)
    high = _with_kwargs(k31_base, {"thinking_effort": "high"}).render(conversation, tools=TOOLS, add_generation_prompt=True)
    assert low.index("thinking_effort=low") < low.index(SYSTEM) and not high.startswith(low[:low.index(SYSTEM)])
    plain = _with_kwargs(k31_base, {"thinking_effort": "low"})
    none = _with_kwargs(k31_base, {"thinking_effort": "low", "tool_choice": "none"})
    history = plain.render(conversation, tools=TOOLS, add_generation_prompt=False)
    assert none.render(conversation, tools=TOOLS, add_generation_prompt=True).startswith(history)
