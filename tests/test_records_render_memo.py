"""Records proxy: one rendering per turn (``_RenderMemo``) gives the same records.

Each scenario runs twice on the same scripted engine: with the reference
``_prompt`` (the implementation before the memo: it renders the previous
turn's history again every turn) and with the current one. The token stream,
loss mask, log-probs, per-turn records, counters, the prompt ids the engine
got, the chat responses and any error must be equal. Tokenizers come from the
local HF cache (tests/hf_assets.py; skipped if absent); the Kimi-K3.1 case
needs ``YETO_K31_TOKENIZER`` (as tests/test_kimi_k31_template.py).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import os
from collections.abc import Callable
from typing import Any

import pytest
from test_records_proxy import (
    SCRIPT,
    SIGNED,
    ScriptedModel,
    _compaction_proxy,
    _template,
    assistant,
    call,
    engine_for,
    run_codex,
)

from yeto.rl.engine.sampler_engine import FINISH_LENGTH
from yeto.rl.harness.chat_templates import ChatTemplate, first_difference
from yeto.rl.harness.gateway import records as records_module
from yeto.rl.harness.gateway.records import RecordingBackend, RecordsMismatch, _first_difference, _Prompt, _same

CHAT_TOOLS = [{"type": "function", "function": {"name": "shell", "description": "run a command", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def reference_prompt(self, session, segment, context):
    """``RecordingBackend._prompt`` before the render memo (verbatim, but for the
    drift / prefill counts: they are flags on the prompt, counted by ``_sample``
    when it samples the prompt)."""
    tools = self.tools.get(session.trajectory_id)
    full = self.template.render(context, tools=tools, add_generation_prompt=True)
    if not segment.turns:
        return _Prompt(stop=(), tail=tuple(self.template.encode(full)))
    if segment.last_finish == FINISH_LENGTH:
        raise RecordsMismatch("cannot continue a stream after a length-truncated turn")
    if len(context) < segment.n_context:
        raise RecordsMismatch("history is shorter than the recorded stream")
    start = self.template.render(context[: segment.n_context - 1], tools=tools, add_generation_prompt=True)
    prefill_dropped = False
    if not full.startswith(start):
        head = self.template.render(context[: segment.n_context - 1], tools=tools, add_generation_prompt=False)
        if not (start.startswith(head) and full.startswith(head)):
            raise RecordsMismatch("the rendering of the earlier turns changed (template rewrites history)")
        start = head + records_module._common_prefix(start[len(head):], full[len(head):])
        prefill_dropped = True
    stop_ids: tuple[int, ...] = ()
    if segment.last_stop is None:
        found = sorted((full.find(s, len(start)), s) for s in self.template.spec.stop_tokens)
        found = [(pos, s) for pos, s in found if pos >= 0]
        if not found:
            raise RecordsMismatch("no stop token after the last assistant turn in the rendering")
        pos, stop = found[0]
        stop_ids = tuple(self.template.encode(stop))
    else:
        stop = segment.last_stop
        pos = full.find(stop, len(start))
        if pos < 0:
            raise RecordsMismatch(f"the rendering has no {stop!r} after the last assistant turn")
    tail = tuple(self.template.encode(full[pos + len(stop):]))
    diff = None
    if self.check_render:
        diff = first_difference([*segment.token_ids, *stop_ids, *tail], self.template.encode(full))
    return _Prompt(stop=stop_ids, tail=tail, render_diff=diff, prefill_dropped=prefill_dropped)


@pytest.fixture(scope="module")
def qwen() -> ChatTemplate:
    return _template("qwen3")


@pytest.fixture(scope="module")
def glm() -> ChatTemplate:
    return _template("glm5")


@pytest.fixture(scope="module")
def qwen3_2507() -> ChatTemplate:
    return _template("qwen3_2507")


def both(monkeypatch, scenario: Callable[[], Any]) -> tuple[Any, Any]:
    """``scenario()`` with the reference ``_prompt``, then with the current one."""
    with monkeypatch.context() as patch:
        patch.setattr(RecordingBackend, "_prompt", reference_prompt)
        reference = scenario()
    return reference, scenario()


def chat_loop(template: ChatTemplate, model: Any, *, turns: int, check_render: bool = True,
              between: Callable[[list[dict[str, Any]], int], None] | None = None, share: bool = False,
              renders: list[int] | None = None) -> dict[str, Any]:
    """Drive ``RecordingBackend.chat`` directly: assistant, tool output, repeat.
    ``between(messages, turn)`` edits the history before the next call;
    ``share``: pass the agent's own message objects (no copy), as a caller that
    keeps and mutates them."""
    engine = engine_for(model)
    backend = RecordingBackend(engine, template, check_render=check_render)
    if renders is not None:
        render = template.render

        def counted(*args, **kw):
            renders.append(1)
            return render(*args, **kw)

        backend.template = copy.copy(template)
        backend.template.render = counted
    backend.declare_tools("t", CHAT_TOOLS)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You are a coding agent."},
                                      {"role": "user", "content": "список files, then read a.txt"}]
    out: dict[str, Any] = {"responses": [], "error": None}

    async def go() -> None:
        sid = await backend.create_session("t")
        for turn in range(turns):
            try:
                response = await backend.chat(sid, messages if share else copy.deepcopy(messages), SIGNED)
            except Exception as exc:  # noqa: BLE001 - the same error on both sides
                out["error"] = (type(exc).__name__, str(exc))
                return
            out["responses"].append(response)
            message = response["choices"][0]["message"]
            messages.append(message)
            for c in message.get("tool_calls") or []:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"output {turn}\n" * 3})
            if between is not None:
                between(messages, turn)

    asyncio.run(go())
    out["records"] = backend.records("t")
    out["prompts"] = [p for p, *_ in engine.token_requests]
    return out


def steps(n: int) -> list[dict[str, Any]]:
    return [assistant(f"step {i}: look", calls=[call(f"c{i}", f"cat file_{i}.txt")]) for i in range(n - 1)] + \
           [assistant("done", content="All read.")]


# ---------------------------------------------------------------------- unit
def test_same_compares_types_and_key_order():
    assert _same({"a": [1, "x", None, 1.5]}, {"a": [1, "x", None, 1.5]})
    assert not _same(True, 1) and not _same(1, 1.0) and not _same((1,), [1])
    assert not _same(-0.0, 0.0) and _same(float("nan"), float("nan"))  # repr: renders the same
    assert not _same({"a": 1, "b": 2}, {"b": 2, "a": 1})  # a template may iterate keys in order
    assert not _same({"a": 1}, {"a": 1, "b": 2}) and not _same([1], [1, 2])

    class Str(str):
        pass

    assert not _same(Str("x"), "x") and not _same(object(), object())


def test_first_difference_is_the_template_one():
    cases = [([], []), ([1], []), ([1, 2], [1, 2]), ([1, 2], [1, 3]), ([1, 2, 3], [1, 2]), ([1], (1,)), ([5], [6, 7])]
    for a, b in cases:
        assert _first_difference(a, b) == first_difference(a, b), (a, b)


# ---------------------------------------------------------------------- parity
@pytest.mark.parametrize("name", ["qwen3", "glm5", "qwen3_2507"])
def test_a_long_tool_loop_records_the_same_stream(name, request, monkeypatch):
    template = request.getfixturevalue({"qwen3": "qwen", "glm5": "glm", "qwen3_2507": "qwen3_2507"}[name])
    script = steps(12)
    ref, new = both(monkeypatch, lambda: chat_loop(template, ScriptedModel(template, script), turns=12))
    assert new["error"] is None and len(new["responses"]) == 12
    assert new == ref
    (session,) = new["records"]["sessions"]
    assert len(session["samples"]) == 1 and new["records"]["counters"].get("render_drift_turns", 0) == 0


def test_one_rendering_per_turn(qwen, monkeypatch):
    renders: dict[str, list[int]] = {"ref": [], "new": []}
    keys = iter(("ref", "new"))
    ref, new = both(monkeypatch, lambda: chat_loop(qwen, ScriptedModel(qwen, steps(8)), turns=8,
                                                   renders=renders[next(keys)]))
    assert new == ref
    # the reference renders the previous history again on every turn after the first
    assert len(renders["ref"]) == 1 + 2 * 7 and len(renders["new"]) == 8


def test_check_render_off_and_a_stop_token_the_engine_does_not_return(glm, monkeypatch):
    for check in (True, False):
        def scenario(check: bool = check) -> dict[str, Any]:
            return chat_loop(glm, ScriptedModel(glm, steps(6), return_stop=False), turns=6, check_render=check)

        ref, new = both(monkeypatch, scenario)
        assert new == ref and new["records"]["counters"]["stop_added"] == 5


def test_render_drift_is_counted_the_same(qwen, monkeypatch):
    """The engine samples a non-canonical tokenization (one character per token):
    the stream trains those ids and both sides count the same drift positions."""

    class Spelled(ScriptedModel):
        def __call__(self, prompt, index):
            ids = super().__call__(prompt, index)
            if index == 0 and len(self.replies) == 3:  # the third turn
                text = self.template.decode(ids[:3])
                ids = [i for ch in text for i in self.template.encode(ch)] + ids[3:]
            return ids

    ref, new = both(monkeypatch, lambda: chat_loop(qwen, Spelled(qwen, steps(6)), turns=6))
    assert new == ref
    counters = new["records"]["counters"]
    assert counters["render_drift_turns"] >= 1
    turns = new["records"]["sessions"][0]["samples"][0]["turns"]
    assert any(t["render_diff"] is not None for t in turns)


def test_a_prefill_the_history_omits(qwen, monkeypatch):
    template = ChatTemplate(dataclasses.replace(qwen.spec, template_kwargs={"enable_thinking": False}), qwen.tokenizer)
    script = [assistant("", calls=[call("c0", "ls")]), assistant("", calls=[call("c1", "cat a")]),
              assistant("", content="done")]
    ref, new = both(monkeypatch, lambda: chat_loop(template, ScriptedModel(template, script), turns=3))
    assert new == ref and new["records"]["counters"]["prefill_dropped_turns"] == 2


@pytest.mark.parametrize("share", [False, True])
def test_an_edited_or_mutated_history_is_rendered_again(glm, monkeypatch, share):
    """The agent edits an earlier tool output: in a fresh copy of the history,
    or in place in message objects the backend saw (``share``). The memo holds
    its own copy, so it is never reused for the edited history: the outcome is
    the reference one (a drift on the edited text)."""

    def edit(messages, turn):
        if turn == 2:
            messages[3]["content"] = "rewritten output\n"

    ref, new = both(monkeypatch, lambda: chat_loop(glm, ScriptedModel(glm, steps(6)), turns=6, between=edit,
                                                   share=share))
    assert new == ref
    assert new["records"]["counters"].get("render_drift_turns", 0) >= 1 or new["error"] is not None


def test_a_rewritten_assistant_turn_counts_the_same_drift(glm, monkeypatch):
    """The agent rewrites an earlier sampled turn: the stream keeps the sampled
    ids (what trains), the fresh rendering differs, and both sides count it."""

    def edit(messages, turn):
        if turn == 1:
            messages[2] = {**messages[2], "reasoning_content": "something else"}

    ref, new = both(monkeypatch, lambda: chat_loop(glm, ScriptedModel(glm, steps(5)), turns=5, between=edit))
    assert new == ref and new["error"] is None
    assert new["records"]["counters"]["render_drift_turns"] == 3  # turns 2..4 see the rewritten turn


def test_a_shorter_history_and_a_type_change_fail_or_render_the_same(glm, monkeypatch):
    def shorten(messages, turn):
        if turn == 1:
            del messages[-2:]

    ref, new = both(monkeypatch, lambda: chat_loop(glm, ScriptedModel(glm, steps(4)), turns=4, between=shorten))
    assert new == ref and new["error"] == ("RecordsMismatch", "history is shorter than the recorded stream")

    def retype(messages, turn):  # same text through json, another Python value: rendered again
        if turn == 1:
            messages[0] = {"role": "system", "content": "You are a coding agent.", "extra": True}

    ref, new = both(monkeypatch, lambda: chat_loop(glm, ScriptedModel(glm, steps(4)), turns=4, between=retype))
    assert new == ref


def test_compaction_records_the_same_segments(glm, monkeypatch):
    def scenario():
        proxy, engine = _compaction_proxy(glm)
        outputs = run_codex(proxy, "t")
        return [[{k: v for k, v in item.items() if k != "id"} for item in o["output"]] for o in outputs], \
            proxy.records("t"), [p for p, *_ in engine.token_requests]

    ref, new = both(monkeypatch, scenario)
    assert new == ref and new[1]["counters"]["compactions"] == 1


def test_codex_responses_loop_records_the_same(qwen3_2507, monkeypatch):
    from yeto.rl.harness.gateway.proxy import RecordsProxy

    def scenario():
        engine = engine_for(ScriptedModel(qwen3_2507, SCRIPT))
        proxy = RecordsProxy(engine, qwen3_2507, model="policy", signed_sampling=SIGNED, max_chains=1)
        outputs = run_codex(proxy, "t1")
        return [[{k: v for k, v in item.items() if k != "id"} for item in o["output"]] for o in outputs], \
            proxy.records("t1"), [p for p, *_ in engine.token_requests]

    ref, new = both(monkeypatch, scenario)
    assert new == ref


# ---------------------------------------------------------------------- Kimi-K3.1
K31 = os.environ.get("YETO_K31_TOKENIZER", "")


class RenderedModel:
    """Replies with the template's own ids for the next scripted assistant
    message (the segment-encoded rendering after a generation prompt, through
    the stop token): what a model that writes the canonical XTML samples."""

    def __init__(self, template: ChatTemplate, replies: list[dict[str, Any]]) -> None:
        self.template, self.replies = template, list(replies)
        self.base = [{"role": "user", "content": "x"}]

    def __call__(self, prompt, index):
        message = self.replies.pop(0)
        head = self.template.encode(self.template.render(self.base, tools=CHAT_TOOLS, add_generation_prompt=True))
        ids = self.template.encode(self.template.render(self.base + [message], tools=CHAT_TOOLS,
                                                        add_generation_prompt=False))
        assert ids[:len(head)] == head
        rest = ids[len(head):]
        cut = next(i for i, t in enumerate(rest) if t in self.template.stop_token_ids)
        return rest[:cut + 1]


@pytest.mark.skipif(not K31 or not os.path.isfile(os.path.join(K31, "tiktoken.model")),
                    reason="set YETO_K31_TOKENIZER to the K3.1 tokenizer files")
def test_kimi_k31_long_loop_records_the_same_stream(monkeypatch):
    pytest.importorskip("tiktoken")
    template = ChatTemplate.load("kimi_k31", tokenizer_path=K31, trust_remote_code=True)
    ref, new = both(monkeypatch, lambda: chat_loop(template, RenderedModel(template, steps(10)), turns=10))
    assert new["error"] is None and new == ref
    assert new["records"]["counters"].get("render_drift_turns", 0) == 0
