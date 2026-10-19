"""Records backend: the gateway ``SessionBackend`` that samples a ``SamplerEngine``
directly and records every token (S25 N10.5).

The gateway (:class:`.core.Gateway`) keeps doing what it does for the Miles
session server: Responses/Chat/Messages -> canonical chat, chain location,
signed sampling. This backend replaces the session server behind it:

* one backend session = one append-only token stream (TITO: token in, token
  out). The first turn renders the chat template; a later turn appends only the
  ids of the new messages after the tokens the engine generated, so the
  recorded ids are the ids the engine saw and sampled;
* per token: id, loss mask (1 = sampled by the policy), log-probability, and
  per turn the policy version and the source (:class:`Provenance`);
* N10.6 drift check: each turn compares the stream with a fresh tokenization of
  the full rendering and records the first differing position. The stream is
  what trains; a drift means the template does not reproduce the generated
  text (counted, not fatal);
* optional context compaction (CompactionRL, :mod:`yeto.rl.compaction`): when
  ``C - |h| < T_comp`` the policy writes a summary in the current segment
  (trainable), and the next segment starts from ``system + u_resume(S) + last
  k steps``. One training sample per segment, as in the Codex bridge.

The chat completion carries SGLang's ``meta_info.output_token_logprobs``, so a
Miles session server can also sit behind this backend.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.compaction import RESUME_TEMPLATE, SUMMARY_PROMPT, CompactionConfig, extract_summary, should_compact
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    Completion,
    Provenance,
    SamplingParams,
    TokenSamplerEngine,
)
from yeto.rl.harness.chat_templates import ChatTemplate, first_difference

from .core import GatewayError

Message = dict[str, Any]

KIND_ACTION = "action"
KIND_SUMMARY = "summary"


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
    stream differs from a fresh tokenization (None: aligned)."""

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
    ``render_diff``: N10.6 drift position (None: aligned or not checked)."""

    stop: tuple[int, ...]
    tail: tuple[int, ...]
    render_diff: int | None = None

    def __len__(self) -> int:
        return len(self.stop) + len(self.tail)


@dataclass
class _Session:
    trajectory_id: str
    segments: list[Segment]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turns: int = 0


def _atomic_step_starts(messages: Sequence[Message]) -> list[int]:
    """Indexes of the assistant messages (a step = assistant + its tool outputs)."""
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


class RecordingBackend:
    """``SessionBackend`` over a :class:`TokenSamplerEngine` (module doc)."""

    def __init__(self, engine: TokenSamplerEngine, template: ChatTemplate, *, sampler_id: str = "records-proxy",
                 compaction: CompactionConfig | None = None, check_render: bool = True) -> None:
        if not isinstance(engine, TokenSamplerEngine):
            raise TypeError(f"{type(engine).__name__} has no generate_tokens (TokenSamplerEngine)")
        self.engine = engine
        self.template = template
        self.sampler_id = sampler_id
        self.compaction = compaction
        self.check_render = check_render
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
                segment = session.segments[-1]
                context = segment.head + list(messages[segment.base:])
                pending = self._prompt(session, segment, context)
            completion = await self._sample(session, segment, pending, params, kind=KIND_ACTION)
            segment.n_context = len(context) + 1
            prompt_tokens = len(segment.token_ids) - len(completion.token_ids)
        message = self._parse(completion)
        return self._chat_response(session_id, message, completion, prompt_tokens=prompt_tokens)

    # -------------------------------------------------------------- records
    def records(self, trajectory_id: str) -> dict[str, Any]:
        sessions = [(sid, s) for sid, s in self.sessions.items() if s.trajectory_id == trajectory_id]
        return {"trajectory_id": trajectory_id, "template": self.template.spec.name,
                "template_verified": self.template.spec.verified,
                "sessions": [{"session_id": sid, "samples": [seg.sample() for seg in s.segments]} for sid, s in sessions],
                "counters": dict(self.counters.get(trajectory_id, Counter()))}

    # -------------------------------------------------------------- internals
    def _prompt(self, session: _Session, segment: Segment, context: list[Message]) -> _Prompt:
        """What to append before the next sample (the whole prompt on a segment's first turn)."""
        tools = self.tools.get(session.trajectory_id)
        full = self.template.render(context, tools=tools, add_generation_prompt=True)
        if not segment.turns:
            return _Prompt(stop=(), tail=tuple(self.template.encode(full)))
        if segment.last_finish == FINISH_LENGTH:
            raise RecordsMismatch("cannot continue a stream after a length-truncated turn")
        if len(context) < segment.n_context:
            raise RecordsMismatch("history is shorter than the recorded stream")
        start = self.template.render(context[: segment.n_context - 1], tools=tools, add_generation_prompt=True)
        if not full.startswith(start):
            raise RecordsMismatch("the rendering of the earlier turns changed (template rewrites history)")
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
            diff = first_difference([*segment.token_ids, *stop_ids, *tail], self.template.encode(full))
            if diff is not None:
                self.counters[session.trajectory_id]["render_drift_turns"] += 1
        return _Prompt(stop=stop_ids, tail=tail, render_diff=diff)

    def _should_compact(self, session: _Session, segment: Segment, pending: _Prompt) -> bool:
        cfg = self.compaction
        if cfg is None or len(session.segments) > cfg.max_compactions:
            return False
        if not any(t.kind == KIND_ACTION for t in segment.turns):
            return False  # no step to summarize yet
        return should_compact(len(segment.token_ids) + len(pending), cfg)

    async def _compact(self, session: _Session, segment: Segment, context: list[Message],
                       params: SamplingParams) -> None:
        """Summary turn in this segment, then open the next one (CompactionRL eq. 8/9).

        ``history`` is the agent's own messages in this segment's context
        (``messages[segment.base:]``); the next segment keeps the last ``k``
        whole steps (assistant + its tool outputs), ``k`` reduced while the
        rebuilt context would trigger again (as :func:`yeto.rl.compaction.rebuild_context`)."""
        cfg = self.compaction
        assert cfg is not None
        summary_context = context + [{"role": "user", "content": SUMMARY_PROMPT}]
        completion = await self._sample(session, segment, self._prompt(session, segment, summary_context),
                                        params, kind=KIND_SUMMARY)
        summary, segment.summary_ok = extract_summary(self.template.decode(self._strip_stop(completion.token_ids)))
        self.counters[session.trajectory_id]["compactions"] += 1

        systems = list(itertools.takewhile(lambda m: m.get("role") == "system", context))
        head = systems + [{"role": "user", "content": RESUME_TEMPLATE.format(summary=summary)}]
        history = context[len(segment.head):]
        steps = _atomic_step_starts(history)
        tools = self.tools.get(session.trajectory_id)
        kept, first = 0, len(history)
        for k in range(min(cfg.keep_recent_steps, len(steps)), 0, -1):
            rebuilt = head + history[steps[-k]:]
            if not should_compact(len(self.template.encode(self.template.render(rebuilt, tools=tools, add_generation_prompt=True))), cfg):
                kept, first = k, steps[-k]
                break
        session.segments.append(Segment(len(session.segments), head=head, base=segment.base + first, kept_steps=kept))

    async def _sample(self, session: _Session, segment: Segment, pending: _Prompt, params: SamplingParams,
                      *, kind: str) -> Completion:
        prompt = [*segment.token_ids, *pending.stop, *pending.tail]
        room = int(self.engine.max_model_len) - len(prompt)
        if room < 1:
            raise ContextFull(f"prompt has {len(prompt)} tokens; max_model_len is {self.engine.max_model_len}")
        params = params.replace(max_tokens=min(params.max_tokens, room))
        version = self.engine.served_version()
        (group,) = await asyncio.to_thread(self.engine.generate_tokens, [prompt], params,
                                           stop_token_ids=self.template.stop_token_ids)
        if self.engine.served_version() != version:  # a reload during the call: version unknown
            raise RecordsMismatch(f"served version changed from {version} during the turn; "
                                  "the turn's behavior version is unknown")
        completion = group[0]
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
        segment.turns.append(TurnRecord(
            turn=session.turns, kind=kind, prompt_added=pending.tail, completion=tuple(completion.token_ids),
            logprobs=tuple(completion.logprobs), finish_reason=completion.finish_reason,
            policy_version=version, provenance=provenance.to_record(), render_diff=pending.render_diff))
        session.turns += 1
        return completion

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
