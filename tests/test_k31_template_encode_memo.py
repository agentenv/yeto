"""Kimi-K3.1 segment encoding with the per-segment ids memo and the per-thread
rendering list: ids equal the uncached segment-by-segment encoding token for
token, the memo stays under its byte bound, and ``encode`` only reads the
calling thread's newest renderings (an ambiguous tail is refused).

Needs the checkpoint's tokenizer (``YETO_K31_TOKENIZER``, as
``tests/test_kimi_k31_template.py``; skipped without it).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
from typing import Any

import pytest

from yeto.rl.harness.chat_templates import ChatTemplate, TemplateError

TOKENIZER = os.environ.get("YETO_K31_TOKENIZER", "")
pytestmark = pytest.mark.skipif(not TOKENIZER or not os.path.isfile(os.path.join(TOKENIZER, "tiktoken.model")),
                                reason="set YETO_K31_TOKENIZER to the K3.1 tokenizer files")

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}, "required": ["command"]}}}]
MARKERS = "<|open|>message role=\"user\"<|sep|>x<|close|>message<|sep|><|end_of_msg|>"


@pytest.fixture(scope="module")
def base() -> ChatTemplate:
    return ChatTemplate.load("kimi_k31", tokenizer_path=TOKENIZER, trust_remote_code=True)


def fresh(base: ChatTemplate, **kwargs: Any) -> ChatTemplate:
    spec = dataclasses.replace(base.spec, template_kwargs={**base.spec.template_kwargs, **kwargs})
    return ChatTemplate(spec, base.tokenizer)


def reference(template: ChatTemplate, messages: list[dict[str, Any]], *, tools: Any = TOOLS,
              add_generation_prompt: bool) -> tuple[str, list[list[int]], list[int]]:
    """(text, ids, segment start offsets) straight from ``build_chat_segments`` and
    ``_encode_text_piece``, segment by segment, with nothing kept (the encoding before the memo)."""
    module = sys.modules[type(template.tokenizer).__module__]
    kwargs = dict(template.spec.template_kwargs)
    thinking = bool(kwargs.pop("thinking", True))
    kwargs.setdefault("thinking_effort", "max")
    segments = module.build_chat_segments(template._ready(messages), tools=list(tools) if tools else None,
                                          add_generation_prompt=add_generation_prompt, thinking=thinking, **kwargs)
    ids, starts, offset = [], [], 0
    for segment in segments:
        starts.append(offset)
        offset += len(segment.text)
        ids.append(list(template.tokenizer._encode_text_piece(segment.text,
                                                              allow_special_tokens=segment.allow_special)))
    return "".join(s.text for s in segments), ids, starts


def flat(pieces: list[list[int]], first: int = 0) -> list[int]:
    return [i for piece in pieces[first:] for i in piece]


def conversation() -> list[dict[str, Any]]:
    long_text = ("word " * 90_000) + ("x" * 30_000) + " Ελληνικά κείμενο " * 2_000  # > 400k chars, long runs
    args = json.dumps({"command": 'grep -n "<|close|>" /app/*.py', "timeout": 30})
    return [
        {"role": "system", "content": "You are pi.\n" + MARKERS},
        {"role": "user", "content": "Fix it. " + MARKERS},
        {"role": "assistant", "content": "", "reasoning_content": "look " + MARKERS,
         "tool_calls": [{"id": "a", "type": "function", "function": {"name": "bash", "arguments": args}},
                        {"id": "b", "type": "function", "function": {"name": "bash", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "second " + MARKERS},
        {"role": "tool", "tool_call_id": "a", "content": long_text},
        {"role": "assistant", "content": "Done. " + MARKERS, "reasoning_content": ""},
        {"role": "user", "content": [{"type": "text", "text": "thanks " + MARKERS}]},
        {"role": "user", "content": "<|end_of_msg|>"},  # plain text equal to a control segment's text
    ]


def check_prefixes(template: ChatTemplate, messages: list[dict[str, Any]]) -> int:
    """Every prefix of the conversation, with and without the generation prompt: the
    rendering and every tail at a segment boundary encode to the reference ids."""
    checked = 0
    for n in range(1, len(messages) + 1):
        for gen in (True, False):
            text = template.render(messages[:n], tools=TOOLS, add_generation_prompt=gen)
            ref_text, pieces, starts = reference(template, messages[:n], add_generation_prompt=gen)
            assert text == ref_text
            assert template.encode(text) == flat(pieces)
            for k in range(1, len(starts)):
                assert template.encode(text[starts[k]:]) == flat(pieces, k), (n, gen, k)
                checked += 1
    return checked


def test_ids_equal_the_uncached_segment_encoding(base):
    template = fresh(base, thinking_effort="low")
    messages = conversation()
    assert check_prefixes(template, messages) > 500
    assert template._pieces, "the memo is used"
    assert check_prefixes(template, messages) > 500  # again with every segment in the memo


@pytest.mark.parametrize("kwargs", [{}, {"thinking": False}, {"thinking_effort": "high", "tool_choice": "none"}])
def test_ids_equal_with_other_template_kwargs_and_an_evicting_memo(base, monkeypatch, kwargs):
    template = fresh(base, **kwargs)
    monkeypatch.setattr(template, "PIECE_MEMO_BYTES", 4096)  # evicts on almost every segment
    check_prefixes(template, conversation()[:4])
    assert template._pieces_bytes <= 4096
    assert template._pieces_bytes == sum(sys.getsizeof(t) + sys.getsizeof(ids) + 160
                                         for (t, _), ids in template._pieces.items())


def test_memo_bytes_stay_bounded_and_renderings_are_per_thread(base, monkeypatch):
    template = fresh(base, thinking_effort="low")
    cap = 2 << 20
    monkeypatch.setattr(template, "PIECE_MEMO_BYTES", cap)
    messages = [{"role": "user", "content": "go"}]
    for i in range(60):  # a growing history with distinct ~40 KiB tool outputs
        messages += [{"role": "assistant", "content": "", "reasoning_content": f"step {i}",
                      "tool_calls": [{"id": f"c{i}", "type": "function",
                                      "function": {"name": "bash", "arguments": json.dumps({"command": f"cat {i}"})}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": f"{i} " * 20_000}]
        text = template.render(messages, tools=TOOLS, add_generation_prompt=True)
        assert template._pieces_bytes <= cap
    assert template.encode(text) == flat(reference(template, messages, add_generation_prompt=True)[1])
    assert len(template._renderings()) == template.RECENT_RENDERINGS


def test_records_proxy_streams_are_the_same_with_and_without_the_memo(base):
    """The compaction agent loop of test_k31_compaction_template: every segment's
    ids and mask are the same as with the uncached encoding."""
    import test_k31_compaction_template as loop

    class Uncached(ChatTemplate):
        def _piece(self, text: str, allow_special: bool) -> Any:
            return list(self.tokenizer._encode_text_piece(text, allow_special_tokens=allow_special))

    template = fresh(base, thinking_effort="low")
    plain = Uncached(template.spec, template.tokenizer)
    runs = [loop.run_agent(t, loop._config(t, keep=2, max_compactions=3)) for t in (template, plain)]
    ours, theirs = (r.proxy.records("t") for r in runs)
    assert ours["counters"] == theirs["counters"] and ours["counters"]["compactions"] == 3
    assert [(s["token_ids"], s["loss_mask"]) for s in ours["sessions"][0]["samples"]] == \
        [(s["token_ids"], s["loss_mask"]) for s in theirs["sessions"][0]["samples"]]


def test_a_tail_of_two_renderings_with_different_ids_is_refused(base):
    """Message text that spells the markers: the same tail text is control tokens in
    one rendering and plain text in another. The old shared LRU took the newest
    match; now both are kept renderings of this thread, so the tail is refused."""
    template = fresh(base, thinking_effort="low")
    first = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    text1 = template.render(first, add_generation_prompt=True)
    tail = text1[text1.index('<|close|>message<|sep|><|end_of_msg|><|open|>message role="user"'):]
    second = [{"role": "user", "content": tail[:tail.index("q<|close|>") + 1]}]
    text2 = template.render(second, add_generation_prompt=True)
    assert text2.endswith(tail)
    ref1 = reference(template, first, tools=None, add_generation_prompt=True)
    ref2 = reference(template, second, tools=None, add_generation_prompt=True)
    ids1, ids2 = flat(ref1[1], ref1[2].index(len(text1) - len(tail))), flat(ref2[1], ref2[2].index(len(text2) - len(tail)))
    assert ids1 != ids2
    with pytest.raises(TemplateError, match="two renderings"):
        template.encode(tail)
    for _ in range(template.RECENT_RENDERINGS):  # once the second rendering is out of the list
        template.render(first, add_generation_prompt=True)
    assert template.encode(tail) == ids1


def test_encode_reads_only_the_calling_threads_renderings(base):
    template = fresh(base, thinking_effort="low")
    messages = conversation()[:3]
    text = template.render(messages, tools=TOOLS, add_generation_prompt=True)
    expected = flat(reference(template, messages, add_generation_prompt=True)[1])
    errors: list[BaseException] = []
    results: list[bool] = []

    def other() -> None:
        try:
            template.encode(text)
        except TemplateError as exc:
            errors.append(exc)

    def worker(i: int) -> None:
        convo = messages + [{"role": "tool", "tool_call_id": "a", "content": f"thread {i} " * 500}]
        for _ in range(5):
            rendered = template.render(convo, tools=TOOLS, add_generation_prompt=True)
            results.append(template.encode(rendered) == flat(reference(template, convo, add_generation_prompt=True)[1]))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert len(errors) == 1, "another thread's rendering is not read"
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True] * 20
    assert template.encode(text) == expected
