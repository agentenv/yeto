"""Records backend: the gateway ``SessionBackend`` that samples a ``SamplerEngine``
directly and records every token.

The gateway (:class:`.core.Gateway`) keeps doing what it does for the Miles
session server: Responses/Chat/Messages -> canonical chat, chain location,
signed sampling. This backend replaces the session server behind it:

* one backend session = one append-only token stream (TITO: token in, token
  out). The first turn renders the chat template; a later turn appends only the
  ids of the new messages after the tokens the engine generated, so the
  recorded ids are the ids the engine saw and sampled;
* per token: id, loss mask (1 = sampled by the policy), log-probability, and
  per turn the policy version and the source (:class:`Provenance`);
* Drift check: each turn compares the stream with a fresh tokenization of
  the full rendering and records the first differing position. The stream is
  what trains; a drift means the template does not reproduce the generated
  text (counted, not fatal);
* one rendering per turn: a turn needs the rendering of the history the last
  turn was sampled on, which is the last turn's own rendering. The session
  keeps it (:class:`_RenderMemo`) with a private copy of the messages, and a
  turn reuses it only when the history is the same, type for type
  (:func:`_same`); otherwise it renders again. Rendering is a pure function of
  the messages and tool schemas, so the ids, masks and counters are the ones
  a second rendering gives (a 262k-token Kimi-K3.1 history costs ~250 ms per
  rendering on one core, and the proxy runs it on its event loop);
* optional context compaction (CompactionRL, :mod:`yeto.rl.compaction`): when
  ``C - |h| < T_comp`` the policy writes a summary in the current segment
  (trainable), and the next segment starts from ``system + u_resume(S) + last
  k steps``. One training sample per segment, as in the Codex bridge.
* R3 rows to the training window (``r3_response_len``): an engine returns one
  routing row per stream position (``[len(prompt) + len(completion) - 1, L_moe,
  top_k]``; GBs at a 1M context), but ``join_turns`` keeps only the first
  ``len(first prompt) + max_response_len - 1`` rows of the last turn. With
  ``r3_response_len`` set to that ``max_response_len`` each turn's rows are cut
  to the window (a copy, so the full array is freed) as soon as the call
  returns; the joined rows are the same. A compaction segment is a sample of
  its own: its window starts at its own first prompt.

The chat completion carries SGLang's ``meta_info.output_token_logprobs``, so a
Miles session server can also sit behind this backend.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.compaction import CompactionConfig, extract_summary, should_compact
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    Completion,
    Provenance,
    SamplingParams,
    TokenSamplerEngine,
    completion_version,
    updates_during_generation,
)
from yeto.rl.harness.chat_templates import ChatTemplate, TemplateError, first_difference

from .core import GatewayError

Message = dict[str, Any]

KIND_ACTION = "action"
KIND_SUMMARY = "summary"
# Stands in for the newest messages in a summary request they leave no room in (RecordingBackend._compact).
OMITTED_FROM_SUMMARY = "[output left out: too long for the summary request]"
# counter, and the parse_error after ``context_full: ``, of a summary request that did not fit
COMPACTION_NO_ROOM = "compaction_no_room"


class RecordsMismatch(GatewayError):
    """The stream cannot be continued token-exactly (fail closed)."""


class ContextFull(GatewayError):
    """The prompt leaves no room for a reply within ``max_model_len``."""


def sampling_params(signed: Mapping[str, Any]) -> SamplingParams:
    """Gateway signed sampling (OpenAI field names) -> :class:`SamplingParams`."""
    if int(signed.get("n", 1)) != 1:
        raise GatewayError("the records proxy samples one completion per request (n=1)")
    max_tokens = next((signed[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in signed), None)
    if max_tokens is None:
        raise GatewayError("signed sampling must set max_tokens")
    stop = signed.get("stop") or ()
    return SamplingParams(
        max_tokens=int(max_tokens), temperature=float(signed.get("temperature", 1.0)),
        top_p=float(signed.get("top_p", 1.0)), top_k=int(signed.get("top_k", -1)),
        seed=None if signed.get("seed") is None else int(signed["seed"]),
        stop=(stop,) if isinstance(stop, str) else tuple(stop),
        repetition_penalty=float(signed.get("repetition_penalty", 1.0)),
        frequency_penalty=float(signed.get("frequency_penalty", 0.0)),
        presence_penalty=float(signed.get("presence_penalty", 0.0)),
    ).check()


@dataclass
class TurnRecord:
    """One engine call. ``prompt_added``: mask-0 ids appended before it;
    ``completion``: the sampled ids (mask 1, stop token included when the
    engine returned it). ``stop_added``: a stop token the engine did not
    return, added with mask 0. ``render_diff``: first position where the
    stream differs from a fresh tokenization (None: aligned). ``parse_error``:
    the policy failed after this completion: the completion is not valid
    model output for the template (HTTP 400), or the next prompt no longer
    fits the context (``context_full: ...``, HTTP 409).  The sampled ids stay
    in the stream."""

    turn: int
    kind: str
    prompt_added: tuple[int, ...]
    completion: tuple[int, ...]
    logprobs: tuple[float, ...]
    finish_reason: str
    policy_version: int | None
    provenance: dict[str, str]
    stop_added: tuple[int, ...] = ()
    render_diff: int | None = None
    parse_error: str | None = None
    # engine rollout arrays (None without them): top-K ``[len(completion), K]``,
    # R3 ``[len(prompt) + len(completion) - 1, L_moe, top_k]``; only the newest
    # turn of a segment keeps its R3 (it covers every earlier position)
    top_ids: Any = None
    top_logprobs: Any = None
    routed_experts: Any = None
    routed_experts_cut: bool = False  # only the training window's rows (RecordingBackend.r3_response_len)


@dataclass
class Segment:
    """One training sample. Context = ``head + history[base:]``; ``n_context``:
    context messages through the last generated assistant message."""

    index: int
    head: list[Message]
    base: int
    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    n_context: int = 0
    last_stop: str | None = None  # stop token text of the last turn (None: not returned)
    last_finish: str | None = None
    kept_steps: int | None = None
    summary_ok: bool | None = None

    def extend(self, ids: Sequence[int], *, mask: int, logprobs: Sequence[float] | None = None) -> None:
        self.token_ids.extend(int(t) for t in ids)
        self.loss_mask.extend([mask] * len(ids))
        self.logprobs.extend([0.0] * len(ids) if logprobs is None else [float(x) for x in logprobs])

    def sample(self) -> dict[str, Any]:
        return {"segment": self.index, "token_ids": list(self.token_ids), "loss_mask": list(self.loss_mask),
                "logprobs": list(self.logprobs), "kept_steps": self.kept_steps, "summary_ok": self.summary_ok,
                "turns": [vars(t).copy() for t in self.turns]}


@dataclass(frozen=True)
class _Prompt:
    """Ids to append before the next sample. ``stop``: a stop token the engine
    ended on but did not return (mask 0); ``tail``: the new messages (mask 0);
    ``render_diff``: drift position (None: aligned or not checked)."""

    stop: tuple[int, ...]
    tail: tuple[int, ...]
    render_diff: int | None = None

    def __len__(self) -> int:
        return len(self.stop) + len(self.tail)


@dataclass(frozen=True)
class _RenderMemo:
    """The last generation-prompt rendering of a session: ``text`` =
    ``render(context, tools=tools, add_generation_prompt=True)``; ``context``
    and ``tools`` are private copies (the caller's messages may change later)."""

    context: list[Message]
    tools: list[Message] | None
    text: str

    def leads(self, context: Sequence[Message], tools: Sequence[Message] | None) -> bool:
        """The memo's messages are the first messages of ``context`` and its tools
        are ``tools``, type for type (:func:`_same`)."""
        return (len(self.context) <= len(context) and _same(tools, self.tools)
                and all(_same(a, b) for a, b in zip(self.context, context)))

    @classmethod
    def after(cls, lead: _RenderMemo | None, context: Sequence[Message], tools: Sequence[Message] | None,
              text: str) -> _RenderMemo:
        """The memo of ``context``'s rendering; ``lead`` (one that :meth:`leads` it)
        gives the copies of its messages, so only the new messages are copied."""
        if lead is None:
            return cls(copy.deepcopy(list(context)), copy.deepcopy(None if tools is None else list(tools)), text)
        return cls(lead.context + copy.deepcopy(list(context[len(lead.context):])), lead.tools, text)


_SCALARS = (str, int, bool, type(None), bytes)


def _same(a: Any, b: Any) -> bool:
    """Structural equality that also compares types and dict key order (``True``
    is not ``1``, ``(1,)`` is not ``[1]``, ``-0.0`` is not ``0.0``): equal values
    render to the same text. Other types are never the same (render again)."""
    kind = type(a)
    if kind is not type(b):
        return False
    if kind is dict:
        return len(a) == len(b) and all(_same(ka, kb) and _same(va, vb)
                                        for (ka, va), (kb, vb) in zip(a.items(), b.items()))
    if kind is list or kind is tuple:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if kind is float:
        return repr(a) == repr(b)
    return kind in _SCALARS and a == b


def _first_difference(a: list[int], b: Sequence[int]) -> int | None:
    """:func:`first_difference`, the aligned case (every turn without drift) at C speed."""
    return None if a == b else first_difference(a, b)


@dataclass
class _Session:
    trajectory_id: str
    segments: list[Segment]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turns: int = 0
    render_memo: _RenderMemo | None = None  # the last _prompt rendering (RecordingBackend._prompt)


def _common_prefix(a: str, b: str) -> str:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _atomic_step_starts(messages: Sequence[Message]) -> list[int]:
    """Indexes of the assistant messages (a step = assistant + its tool outputs)."""
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


class RecordingBackend:
    """``SessionBackend`` over a :class:`TokenSamplerEngine` (module doc)."""

    def __init__(self, engine: TokenSamplerEngine, template: ChatTemplate, *, sampler_id: str = "records-proxy",
                 compaction: CompactionConfig | None = None, check_render: bool = True,
                 r3_response_len: int | None = None) -> None:
        if r3_response_len is not None and int(r3_response_len) <= 0:
            raise ValueError("r3_response_len must be positive")
        if not isinstance(engine, TokenSamplerEngine):
            raise TypeError(f"{type(engine).__name__} has no generate_tokens (TokenSamplerEngine)")
        self.engine = engine
        self.template = template
        self.sampler_id = sampler_id
        self.compaction = compaction
        self.check_render = check_render
        self.r3_response_len = None if r3_response_len is None else int(r3_response_len)
        self.sessions: dict[str, _Session] = {}
        self.tools: dict[str, list[Message]] = {}
        self.counters: dict[str, Counter[str]] = {}
        self._ids = itertools.count()

    # -------------------------------------------------------------- gateway seam
    def declare_tools(self, trajectory_id: str, tools: Sequence[Message] | None) -> None:
        """Tool schemas of a trajectory (rendered into the prompt); fixed once set."""
        tools = list(tools or [])
        known = self.tools.setdefault(trajectory_id, tools)
        if known != tools:
            raise RecordsMismatch("tool schemas changed within a trajectory; the rendered prompt would change")

    async def create_session(self, trajectory_id: str) -> str:
        session_id = f"rec{next(self._ids)}"
        self.sessions[session_id] = _Session(trajectory_id, [Segment(0, head=[], base=0)])
        self.counters.setdefault(trajectory_id, Counter())
        return session_id

    async def chat(self, session_id: str, messages: list[Message], sampling: dict[str, Any]) -> dict[str, Any]:
        session = self.sessions[session_id]
        params = sampling_params(sampling)
        async with session.lock:
            segment = session.segments[-1]
            context = segment.head + list(messages[segment.base:])
            pending = self._prompt(session, segment, context)
            if self._should_compact(session, segment, pending):
                await self._compact(session, segment, context, params)
                if self.sessions.get(session_id) is not session:  # forgotten during the summary turn
                    raise RecordsMismatch("the trajectory was forgotten during its summary turn")
                segment = session.segments[-1]
                context = segment.head + list(messages[segment.base:])
                pending = self._prompt(session, segment, context)
            completion = await self._sample(session, segment, pending, params, kind=KIND_ACTION)
            segment.n_context = len(context) + 1
            prompt_tokens = len(segment.token_ids) - len(completion.token_ids)
        try:
            message = self._parse(completion)
        except TemplateError as exc:  # malformed model output: the policy's failure
            segment.turns[-1].parse_error = str(exc)[:500]
            self.counters[session.trajectory_id]["policy_output_errors"] += 1
            raise
        return self._chat_response(session_id, message, completion, prompt_tokens=prompt_tokens)

    # -------------------------------------------------------------- records
    def records(self, trajectory_id: str) -> dict[str, Any]:
        sessions = [(sid, s) for sid, s in self.sessions.items() if s.trajectory_id == trajectory_id]
        return {"trajectory_id": trajectory_id, "template": self.template.spec.name,
                "template_verified": self.template.spec.verified,
                "sessions": [{"session_id": sid, "samples": [seg.sample() for seg in s.segments]} for sid, s in sessions],
                "counters": dict(self.counters.get(trajectory_id, Counter()))}

    def forget(self, trajectory_id: str) -> None:
        """Drop everything recorded for the trajectory (after its records were read)."""
        for session_id in [sid for sid, s in self.sessions.items() if s.trajectory_id == trajectory_id]:
            del self.sessions[session_id]
        self.tools.pop(trajectory_id, None)
        self.counters.pop(trajectory_id, None)

    # -------------------------------------------------------------- internals
    def _prompt(self, session: _Session, segment: Segment, context: list[Message]) -> _Prompt:
        """What to append before the next sample (the whole prompt on a segment's first turn)."""
        tools = self.tools.get(session.trajectory_id)
        memo = session.render_memo
        if memo is not None and not memo.leads(context, tools):
            memo = None  # another history (a new segment, a summary request, an edited history)
        full = self.template.render(context, tools=tools, add_generation_prompt=True)
        try:
            session.render_memo = _RenderMemo.after(memo, context, tools, full)
        except (TypeError, copy.Error, RecursionError):  # not copyable: no memo, the next turn renders again
            session.render_memo = None
        if not segment.turns:
            return _Prompt(stop=(), tail=tuple(self.template.encode(full)))
        if segment.last_finish == FINISH_LENGTH:
            if self.compaction is not None and len(segment.token_ids) >= int(self.engine.max_model_len):
                # cut because the stream reached the engine context, not by the turn's max_tokens:
                # the trajectory ran out of context (as a compacted one that no compaction can help)
                self._context_full(session, segment, f"the stream has {len(segment.token_ids)} tokens after a "
                                                     f"turn cut at max_model_len {self.engine.max_model_len}")
            raise RecordsMismatch("cannot continue a stream after a length-truncated turn")
        if len(context) < segment.n_context:
            raise RecordsMismatch("history is shorter than the recorded stream")
        if memo is not None and len(memo.context) == segment.n_context - 1:
            start = memo.text  # the last turn's rendering of these same messages (module doc)
        else:
            start = self.template.render(context[: segment.n_context - 1], tools=tools, add_generation_prompt=True)
        if not full.startswith(start):
            # The generation prompt may end in a template prefill that the history form of the
            # turn omits (Qwen3 enable_thinking=False: ``<think>\n\n</think>\n\n``). The
            # prefill is template text the engine saw as prompt ids, not sampled text: the stream
            # keeps it, and only the turns before it must render unchanged.
            head = self.template.render(context[: segment.n_context - 1], tools=tools, add_generation_prompt=False)
            if not (start.startswith(head) and full.startswith(head)):
                raise RecordsMismatch("the rendering of the earlier turns changed (template rewrites history)")
            start = head + _common_prefix(start[len(head):], full[len(head):])
            self.counters[session.trajectory_id]["prefill_dropped_turns"] += 1
        stop_ids: tuple[int, ...] = ()
        if segment.last_stop is None:  # the engine ended on a stop token but did not return it
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
            diff = _first_difference([*segment.token_ids, *stop_ids, *tail], self.template.encode(full))
            if diff is not None:
                self.counters[session.trajectory_id]["render_drift_turns"] += 1
        return _Prompt(stop=stop_ids, tail=tail, render_diff=diff)

    def _should_compact(self, session: _Session, segment: Segment, pending: _Prompt) -> bool:
        """Compact before this turn (``C - |h| < T_comp``, a step in the segment to summarize).

        A segment is not compacted again when no compaction is left
        (``max_compactions``) or when it already started over the trigger (its
        rebuilt context, even with no kept step, crossed ``C - T_comp``: another
        summary would only repeat every turn). Such a segment runs until the
        context reaches ``C``, then :class:`ContextFull` ends the trajectory (the
        policy's budget, as :meth:`_sample`'s ``max_model_len``)."""
        cfg = self.compaction
        if cfg is None:
            return False
        if not any(t.kind == KIND_ACTION for t in segment.turns):
            return False  # no step to summarize yet
        used = len(segment.token_ids) + len(pending)
        if not should_compact(used, cfg):
            return False
        opened_over = segment.index > 0 and should_compact(len(segment.turns[0].prompt_added), cfg)
        if len(session.segments) <= cfg.max_compactions and not opened_over:
            return True
        if used >= cfg.context_budget:
            left = "no compaction is left" if not opened_over else "the segment started over the compaction trigger"
            self.counters[session.trajectory_id]["context_budget_full"] += 1
            self._context_full(session, segment, f"prompt has {used} tokens; context_budget is "
                                                 f"{cfg.context_budget} and {left}")
        return False

    def _context_full(self, session: _Session, segment: Segment, message: str) -> None:
        """End the trajectory at a full context (raises :class:`ContextFull`): marked
        ``context_full: ...`` on the segment's last sampled turn, as :meth:`_sample`
        does at ``max_model_len`` (the trajectory ran out of context; not a parse failure)."""
        if segment.turns:
            segment.turns[-1].parse_error = f"context_full: {message}"
        self.counters[session.trajectory_id]["context_full"] += 1
        raise ContextFull(message)

    async def _compact(self, session: _Session, segment: Segment, context: list[Message],
                       params: SamplingParams) -> None:
        """Summary turn in this segment, then open the next one (CompactionRL eq. 8/9).

        ``history`` is the agent's own messages in this segment's context
        (``messages[segment.base:]``); the next segment keeps the last ``k``
        whole steps (assistant + its tool outputs), ``k`` reduced while the
        rebuilt context would trigger again (as :func:`yeto.rl.compaction.rebuild_context`).

        The summary turn samples ``cfg.summary_max_tokens`` (else the turn's
        ``max_tokens`` capped at :data:`~yeto.rl.compaction.DEFAULT_SUMMARY_MAX_TOKENS`).
        When the newest messages leave the engine less room than that (a large
        tool output), they are replaced by a note in the summary request only
        (this segment is not continued); the next segment still
        keeps them verbatim when they fit. The summary is read from the parsed
        reply's content (the response channel; a ``<summary>`` drafted in the
        reasoning does not count); with no content (cut inside the reasoning)
        the reasoning text is kept, ``summary_ok`` False. A reply cut
        at its length, without a closed ``<summary>``, with tool calls (not run)
        or that does not parse still opens the next segment, with ``summary_ok``
        False and a counter."""
        cfg = self.compaction
        assert cfg is not None
        counters = self.counters[session.trajectory_id]
        params = params.replace(max_tokens=cfg.summary_tokens(params.max_tokens))
        request = {"role": "user", "content": cfg.summary_prompt}
        pending = self._prompt(session, segment, context + [request])
        room = int(self.engine.max_model_len) - len(segment.token_ids) - len(pending)
        if room < params.max_tokens and len(context) > segment.n_context:
            newest = [{**m, "content": OMITTED_FROM_SUMMARY} for m in context[segment.n_context:]]
            pending = self._prompt(session, segment, context[: segment.n_context] + newest + [request])
            counters["summary_newest_omitted"] += 1
        prompt_len = len(segment.token_ids) + len(pending)
        if prompt_len >= int(self.engine.max_model_len):  # nothing left to write the summary in
            counters[COMPACTION_NO_ROOM] += 1
            self._context_full(session, segment, f"{COMPACTION_NO_ROOM}: the summary request has {prompt_len} "
                                                 f"tokens; max_model_len is {self.engine.max_model_len}")
        completion = await self._sample(session, segment, pending, params, kind=KIND_SUMMARY)
        if self.counters.get(session.trajectory_id) is not counters:  # forgotten while the engine wrote it
            raise RecordsMismatch("the trajectory was forgotten during its summary turn")
        if completion.finish_reason == FINISH_LENGTH:
            counters["summary_truncated"] += 1
        try:
            reply: Message | None = self._parse(completion)
        except TemplateError:
            reply = None
            counters["summary_parse_errors"] += 1
        from_reasoning = False
        if reply is None:
            text = self.template.decode(self._strip_stop(completion.token_ids))
        else:
            text = str(reply.get("content") or "")
            if not text.strip():  # cut inside the reasoning: what was written (never a good summary)
                text, from_reasoning = str(reply.get("reasoning_content") or ""), True
            if reply.get("tool_calls"):  # against the request; never run, the agent does not see this turn
                counters["summary_tool_calls"] += 1
        summary, ok = extract_summary(text)
        segment.summary_ok = (ok and bool(summary) and reply is not None and not from_reasoning
                              and not reply.get("tool_calls"))
        counters["compactions"] += 1
        if not segment.summary_ok:
            counters["summary_not_ok"] += 1

        systems = list(itertools.takewhile(lambda m: m.get("role") == "system", context))
        head = systems + [{"role": "user", "content": cfg.resume_template.format(summary=summary)}]
        history = context[len(segment.head):]
        steps = _atomic_step_starts(history)
        tools = self.tools.get(session.trajectory_id)

        def tokens(rebuilt: list[Message]) -> int:
            return len(self.template.encode(self.template.render(rebuilt, tools=tools, add_generation_prompt=True)))

        kept, first = 0, len(history)
        for k in range(min(cfg.keep_recent_steps, len(steps)), 0, -1):
            if not should_compact(tokens(head + history[steps[-k]:]), cfg):
                kept, first = k, steps[-k]
                break
        else:
            used = tokens(head)
            if used >= int(self.engine.max_model_len):  # the next segment could not sample a token
                counters["compaction_no_room"] += 1
                self._context_full(session, segment, f"compaction_no_room: the rebuilt context has {used} "
                                                     f"tokens; max_model_len is {self.engine.max_model_len}")
            if should_compact(used, cfg):  # even k = 0 is over the trigger: _should_compact will not compact it again
                counters["compaction_rebuild_over_budget"] += 1
        session.segments.append(Segment(len(session.segments), head=head, base=segment.base + first, kept_steps=kept))

    async def _sample(self, session: _Session, segment: Segment, pending: _Prompt, params: SamplingParams,
                      *, kind: str) -> Completion:
        prompt = [*segment.token_ids, *pending.stop, *pending.tail]
        room = int(self.engine.max_model_len) - len(prompt)
        if room < 1:
            message = f"prompt has {len(prompt)} tokens; max_model_len is {self.engine.max_model_len}"
            # The policy filled its own context (a budget, like a token cap): mark it on the last
            # sampled turn so the outer layer trains it as a 0 instead of an infrastructure abort.
            if segment.turns:
                segment.turns[-1].parse_error = f"context_full: {message}"
            self.counters[session.trajectory_id]["context_full"] += 1
            raise ContextFull(message)
        params = params.replace(max_tokens=min(params.max_tokens, room))
        version = self.engine.served_version()
        (group,) = await asyncio.to_thread(self.engine.generate_tokens, [prompt], params,
                                           stop_token_ids=self.template.stop_token_ids)
        completion = group[0]
        if updates_during_generation(self.engine):  # a live reload may fall inside the call
            version = completion_version(completion, version, live=True)  # its first token's version
        elif self.engine.served_version() != version:  # a reload during the call: version unknown
            raise RecordsMismatch(f"served version changed from {version} during the turn; "
                                  "the turn's behavior version is unknown")
        if completion.prompt_token_ids and tuple(completion.prompt_token_ids) != tuple(prompt):
            raise RecordsMismatch("the engine reports other prompt ids than the proxy sent")
        if pending.stop:
            segment.turns[-1].stop_added = pending.stop
            self.counters[session.trajectory_id]["stop_added"] += 1
        segment.extend(pending.stop, mask=0)
        segment.extend(pending.tail, mask=0)
        segment.extend(completion.token_ids, mask=1, logprobs=completion.logprobs)
        last = completion.token_ids[-1] if completion.token_ids else None
        segment.last_stop = None if last is None else self.template.stop_text(last)
        segment.last_finish = completion.finish_reason
        provenance = Provenance.of(self.engine.identity, sampler_id=self.sampler_id, sampling=params)
        if segment.turns:
            segment.turns[-1].routed_experts = None  # superseded by this turn's rows
            segment.turns[-1].routed_experts_cut = False
        segment.turns.append(TurnRecord(
            top_ids=completion.top_logprobs_ids, top_logprobs=completion.top_logprobs,
            routed_experts=completion.routed_experts,
            turn=session.turns, kind=kind, prompt_added=pending.tail, completion=tuple(completion.token_ids),
            logprobs=tuple(completion.logprobs), finish_reason=completion.finish_reason,
            policy_version=version, provenance=provenance.to_record(), render_diff=pending.render_diff))
        self._cut_routed_experts(session, segment, segment.turns[-1])
        session.turns += 1
        return completion

    def _r3_rows_cap(self, trajectory_id: str, segment: Segment | None = None) -> int | None:
        """Rows of R3 ``join_turns`` can use for the trajectory: its first recorded
        call's prompt + ``r3_response_len`` - 1 (module doc); in a compaction
        segment (index > 0, joined as a sample of its own) the segment's first
        call's prompt instead. None (keep every row) without ``r3_response_len``
        or while that first call is not recorded."""
        if self.r3_response_len is None:
            return None
        if segment is not None and segment.index > 0:
            # a segment's first call appends its whole prompt (no stop token before it)
            return len(segment.turns[0].prompt_added) + self.r3_response_len - 1 if segment.turns else None
        # records() lists sessions in creation order and segments in order: the
        # join's first turn is the first turn of the trajectory's first segment
        first = next((s for s in self.sessions.values() if s.trajectory_id == trajectory_id), None)
        if first is None or not first.segments[0].turns:
            return None
        return len(first.segments[0].turns[0].prompt_added) + self.r3_response_len - 1

    def _cut_routed_experts(self, session: _Session, segment: Segment, record: TurnRecord) -> None:
        rows = record.routed_experts
        cap = self._r3_rows_cap(session.trajectory_id, segment)
        if rows is None or cap is None or len(rows) <= cap:
            return
        record.routed_experts = rows[:cap].copy()  # a copy: the full array is not kept alive
        record.routed_experts_cut = True

    def _strip_stop(self, ids: Sequence[int]) -> list[int]:
        ids = list(ids)
        return ids[:-1] if ids and self.template.stop_text(ids[-1]) is not None else ids

    def _parse(self, completion: Completion) -> Message:
        return self.template.parse(self.template.decode(self._strip_stop(completion.token_ids)))

    @staticmethod
    def _chat_response(session_id: str, message: Message, completion: Completion, *, prompt_tokens: int) -> dict[str, Any]:
        finish = "length" if completion.finish_reason == FINISH_LENGTH else ("tool_calls" if message.get("tool_calls") else "stop")
        n = len(completion.token_ids)
        return {"id": f"chatcmpl-{session_id}", "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish,
                             "meta_info": {"output_token_logprobs": [[lp, tid, None] for lp, tid in
                                                                     zip(completion.logprobs, completion.token_ids)],
                                           "prompt_tokens": prompt_tokens, "completion_tokens": n}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}}
