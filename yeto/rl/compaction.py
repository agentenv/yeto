"""CompactionRL rollout side.

Pure-Python episode driver for compacted agent rollouts, after CompactionRL
(arXiv 2607.05378v1, sec. 4.1 eq. 6-9, sec. 5.1):

* trigger (eq. 7): compact when the remaining context ``C - |h_t| < T_comp``
  (``T_comp`` = 10,240);
* the *same* policy writes the summary ``S_t ~ pi(. | h_t + q_sum)`` (eq. 8),
  with an ``<analysis>`` scratch part and a ``<summary>`` part;
* rebuilt context (eq. 9): ``s + u_resume(S_t) + last k steps`` (k = 2, reduced
  when the rebuilt context would trigger again); a step = (action, observation)
  is atomic (eq. 6). yeto addition (``keep_task_message``, on by default): the
  task message, verbatim, follows ``s`` -- ``s + task + u_resume(S_t) + last k
  steps`` -- so the task survives a compaction as written, not only as the
  summary's paraphrase (cut to ``task_message_max_tokens`` when it would keep
  the rebuilt context over the trigger). Optional (``keep_last_step_max_tokens``,
  off by default): when even k = 1 does not fit, the last step is kept with its
  tool output cut in the middle instead of keeping no step;
* at most 3 compactions per rollout (sec. 5.1); after the last one the episode
  runs until the context is full and is then marked truncated;
* every segment carries the rollout's task reward (shared return, no separate
  summary reward) and segment metadata for cross-segment GAE (eq. 13-15).

The paper does not give the q_sum section names, the ``<analysis>`` wording or
the u_resume text: :data:`SUMMARY_SECTIONS`, :data:`SUMMARY_PROMPT` and
:data:`RESUME_TEMPLATE` are yeto drafts (a 9-section template; the paper's sec. 4.1 lists what must be kept: original goal, completed
actions, important observations, unresolved errors, current state, plausible
next steps) and need user confirmation; a run may replace them
(:attr:`CompactionConfig.summary_prompt` / :attr:`CompactionConfig.resume_template`).

Integration: the driver takes a ``policy(messages) -> str`` and
an ``env`` with ``step(action) -> (observation, done)`` / ``reward()``; wiring it
into the Codex / Terminal-Bench harness or Miles session server is not done.
No torch / miles imports.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol

T_COMP = 10_240
KEEP_RECENT_STEPS = 2
MAX_COMPACTIONS = 3
# Cap of the task message in a rebuilt context, applied only when the whole task
# message would keep the rebuilt context over the trigger (keep_task_message).
TASK_MESSAGE_MAX_TOKENS = 4096
# Least token cap of a cut text (task message, kept tool output): room for the
# truncation marker and some text on both sides of it.
MIN_CUT_TOKENS = 64
# The marker that replaces the middle of a cut text ({n}: tokens left out).
TRUNCATION_MARKER = "\n[... {n} tokens truncated by context compaction ...]\n"
# Tokens reserved for the marker when a text is cut to a token cap.
_MARKER_RESERVE = 24

# yeto draft (paper: not given) -- 9 sections covering the sec. 4.1 list.
SUMMARY_SECTIONS: tuple[str, ...] = (
    "Original task and goal",
    "Constraints and environment facts",
    "Completed actions",
    "Important observations",
    "Files and state changed",
    "Errors and unresolved problems",
    "Current state",
    "Remaining work",
    "Next step",
)

SUMMARY_PROMPT = (
    "Your context is almost full. Summarize the interaction so far so that you can "
    "continue the task from the summary alone.\n"
    "First, inside <analysis></analysis>, note in a few short lines what must not be lost.\n"
    "Then write the summary inside <summary></summary> with exactly these numbered "
    "sections:\n"
    + "".join(f"{i}. {name}\n" for i, name in enumerate(SUMMARY_SECTIONS, 1))
    + "Be brief: a few short lines per section, no restated tool output.\n"
    "Copy verbatim, once each: task goal and constraints, file paths with full "
    "directories, commands, IDs, URLs, key numbers.\n"
    "Do not call any tool in this turn."
)

RESUME_TEMPLATE = (
    "This session continues from an earlier one whose context was compacted. "
    "Summary of the earlier interaction:\n\n{summary}\n\n"
    "If the original task message is shown above, it is authoritative: follow it, and take "
    "file paths, commands and IDs from it verbatim, not from this summary. "
    "The most recent steps follow verbatim. Continue the task."
)

Message = dict[str, str]


class Environment(Protocol):
    def step(self, action: str) -> tuple[str, bool]: ...

    def reward(self) -> float: ...


def default_count_tokens(text: str) -> int:
    """Whitespace token count (tests); real runs pass the tokenizer's count."""

    return len(text.split())


# Cap of the summary turn's max_tokens when no summary_max_tokens is set: the summary
# turn and the next segment's first action are sampled within one agent request, and
# two full 8192-token turns at long-context ITL outlast the agent's HTTP idle timeout.
DEFAULT_SUMMARY_MAX_TOKENS = 4096

# Tokens of the summary request beyond its text: the template's user-message
# wrapper, the assistant generation prompt and a stop token the engine did not
# return (a bound for the shipped templates: GLM, Qwen3, Kimi K3 XTML).
SUMMARY_REQUEST_OVERHEAD = 64


def summary_request_tokens(summary_prompt: str) -> int:
    """Upper bound on the tokens the summary request adds to the context: one
    token per UTF-8 byte of ``summary_prompt`` (a byte-level BPE token is at
    least one byte) plus :data:`SUMMARY_REQUEST_OVERHEAD`. Needs no tokenizer,
    so a spec is checked before any model is loaded."""
    return len(summary_prompt.encode("utf-8")) + SUMMARY_REQUEST_OVERHEAD


@dataclass(frozen=True)
class CompactionConfig:
    """``summary_max_tokens``: the summary turn's ``max_tokens`` (None: the
    agent's signed per-turn ``max_tokens`` capped at
    :data:`DEFAULT_SUMMARY_MAX_TOKENS`); when set, ``t_comp`` must hold it plus
    the summary request (:func:`summary_request_tokens`).
    ``summary_prompt`` / ``resume_template``: q_sum and u_resume (default: the
    yeto drafts :data:`SUMMARY_PROMPT` / :data:`RESUME_TEMPLATE`);
    ``resume_template`` has one ``{summary}`` field.

    ``keep_task_message`` (default True): a rebuilt context keeps the
    trajectory's first user message (the task) verbatim after the system
    messages and before u_resume; False is the paper's ``s + u_resume + last k
    steps``. ``task_message_max_tokens``: when the whole task message would
    keep the rebuilt context over the trigger, it is cut to about this many
    tokens (head and tail kept, :data:`TRUNCATION_MARKER` in the middle).
    ``keep_last_step_max_tokens`` (default None: off): when not even one whole
    step fits (k would be 0), keep the last step with its tool output cut to
    about this many tokens in total (the policy's call is never cut)."""

    context_budget: int  # C (paper: 64k GLM-4.7-Flash, 80k GLM-4.5-Air-SFT)
    t_comp: int = T_COMP
    keep_recent_steps: int = KEEP_RECENT_STEPS
    max_compactions: int = MAX_COMPACTIONS
    max_turns: int = 250  # evaluation turn cap (sec. 5.1); training value not given
    summary_max_tokens: int | None = None
    summary_prompt: str = SUMMARY_PROMPT
    resume_template: str = RESUME_TEMPLATE
    keep_task_message: bool = True
    task_message_max_tokens: int = TASK_MESSAGE_MAX_TOKENS
    keep_last_step_max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.context_budget <= self.t_comp:
            raise ValueError("context_budget must exceed t_comp")
        if self.keep_recent_steps < 0 or self.max_compactions < 0 or self.max_turns < 1:
            raise ValueError("keep_recent_steps/max_compactions must be >= 0, max_turns >= 1")
        if not isinstance(self.summary_prompt, str) or not self.summary_prompt.strip():
            raise ValueError("summary_prompt must be a non-empty string")
        if not isinstance(self.resume_template, str) or "{summary}" not in self.resume_template:
            raise ValueError("resume_template must be a string with a {summary} field")
        try:
            self.resume_template.format(summary="")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"resume_template may have only the {{summary}} field "
                             f"(write other braces as {{{{ }}}}): {exc!r}") from exc
        if self.summary_max_tokens is not None:
            if isinstance(self.summary_max_tokens, bool) or not isinstance(self.summary_max_tokens, int):
                raise ValueError(f"summary_max_tokens must be an integer, got {self.summary_max_tokens!r}")
            self.check_summary_room(self.summary_max_tokens)
        if not isinstance(self.keep_task_message, bool):
            raise ValueError(f"keep_task_message must be true or false, got {self.keep_task_message!r}")
        for name in ("task_message_max_tokens", "keep_last_step_max_tokens"):
            value = getattr(self, name)
            if value is None and name == "keep_last_step_max_tokens":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < MIN_CUT_TOKENS:
                raise ValueError(f"{name} must be >= {MIN_CUT_TOKENS} (room for the truncation marker), "
                                 f"got {value}")

    def summary_tokens(self, turn_max_tokens: int) -> int:
        """The summary turn's ``max_tokens`` given the signed per-turn one."""
        if self.summary_max_tokens is None:
            return min(int(turn_max_tokens), DEFAULT_SUMMARY_MAX_TOKENS)
        return int(self.summary_max_tokens)

    def check_summary_room(self, turn_max_tokens: int) -> int:
        """Refuse (ValueError) a ``t_comp`` that cannot hold the summary turn:
        compaction starts once fewer than ``t_comp`` tokens remain, and that
        room must take the summary request and a summary of
        :meth:`summary_tokens`. Returns those tokens. A spec reader calls it
        with the signed per-turn ``max_tokens`` (``agentic.turn_max_tokens``)."""
        tokens = self.summary_tokens(turn_max_tokens)
        if tokens < 1:
            raise ValueError(f"the summary turn's max_tokens must be >= 1, got {tokens}")
        need = tokens + summary_request_tokens(self.summary_prompt)
        if need > self.t_comp:
            raise ValueError(
                f"t_comp {self.t_comp} cannot hold the summary turn: max_tokens {tokens} + up to "
                f"{need - tokens} tokens of summary request = {need}; raise t_comp or set "
                f"summary_max_tokens <= {self.t_comp - (need - tokens)}")
        return tokens

    def check_summary_headroom(self, max_model_len: int, turn_max_tokens: int) -> int:
        """Refuse (ValueError) a budget whose summary turn can overrun the engine context.

        The step before a compaction starts below the trigger (at most
        ``C - t_comp`` tokens), adds one reply (``turn_max_tokens``) and then
        tool output of any length. The records proxy leaves that newest output
        out of the summary request when it does not fit, so the summary turn
        always fits within ``max_model_len`` when
        ``C - t_comp + turn_max_tokens + summary request + summary <= max_model_len``.
        Returns the slack (tokens to spare). A spec reader calls it with the
        engine's ``max_model_len`` and ``agentic.turn_max_tokens``."""
        summary = self.summary_tokens(turn_max_tokens)
        need = (self.context_budget - self.t_comp + int(turn_max_tokens)
                + summary_request_tokens(self.summary_prompt) + summary)
        if need > int(max_model_len):
            over = need - int(max_model_len)
            raise ValueError(
                f"the summary turn can overrun max_model_len {max_model_len} by {over} tokens: "
                f"context_budget {self.context_budget} - t_comp {self.t_comp} + turn_max_tokens "
                f"{turn_max_tokens} + summary request {summary_request_tokens(self.summary_prompt)} + "
                f"summary {summary} = {need}; lower context_budget to <= {self.context_budget - over}, "
                f"raise t_comp, or lower summary_max_tokens")
        return int(max_model_len) - need


def context_tokens(messages: Sequence[Message], count: Callable[[str], int]) -> int:
    return sum(count(m["content"]) for m in messages)


def should_compact(used_tokens: int, cfg: CompactionConfig) -> bool:
    """Eq. 7: ``C - |h_t| < T_comp``."""

    return cfg.context_budget - used_tokens < cfg.t_comp


def extract_summary(text: str) -> tuple[str, bool]:
    """``(summary, ok)``: the ``<summary>`` body, ``<analysis>`` dropped.

    Without a closed ``<summary>`` tag ``ok`` is False (recorded in the
    segment metadata) and the text kept is: after an unclosed ``<summary>``
    (a reply cut at its length), the part written; otherwise the text minus
    the closed analysis blocks, or an unclosed analysis's own text when
    nothing else is left.
    """

    m = re.search(r"<summary>(.*?)</summary>", text, flags=re.S)
    if m:
        return m.group(1).strip(), True
    stripped = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.S)
    if "<summary>" in stripped:  # cut inside the summary (length): the part written
        return stripped.rsplit("<summary>", 1)[1].strip(), False
    if "<analysis>" in stripped:  # cut inside the analysis: its text, else nothing would be kept
        head, scratch = stripped.split("<analysis>", 1)
        stripped = head if head.strip() else scratch
    return stripped.strip(), False


# Ends a summary that was cut at its length (:func:`salvage_summary`).
SUMMARY_CUT_NOTE = "[The summary was cut at its length limit; later sections may be missing.]"
_SECTION_HEAD = re.compile(r"^\s*(?:#+\s*|\*\*)?([1-9])[.)]\s", flags=re.M)


def salvage_summary(text: str) -> tuple[str, int]:
    """``(summary, sections)``: best-effort summary of a summary turn cut at its
    length without a closed ``<summary>`` tag. The text :func:`extract_summary`
    keeps, from the first numbered section header on when there is one, without
    its last line (cut mid-way: a half-written path is worse than none), and
    :data:`SUMMARY_CUT_NOTE` at the end. ``sections``: numbered section headers
    kept (0: none, the text is kept as written)."""
    body, _ok = extract_summary(text)
    lines = body.splitlines()
    if len(lines) > 1:
        lines = lines[:-1]
    body = "\n".join(lines).rstrip()
    first = _SECTION_HEAD.search(body)
    if first is not None:
        body = body[first.start():].lstrip("\n")
    sections = len({m.group(1) for m in _SECTION_HEAD.finditer(body)})
    return (f"{body}\n{SUMMARY_CUT_NOTE}" if body else SUMMARY_CUT_NOTE), sections


def step_messages(step: tuple[str, str]) -> list[Message]:
    action, observation = step
    return [{"role": "assistant", "content": action}, {"role": "user", "content": observation}]


def truncate_middle(text: str, max_tokens: int, encode: Callable[[str], Sequence[Any]],
                    decode: Callable[[Sequence[Any]], str]) -> tuple[str, int]:
    """``(text, tokens left out)``: ``text`` cut to about ``max_tokens`` tokens,
    its head and tail kept and :data:`TRUNCATION_MARKER` in the middle; the
    text unchanged (0) when it already fits. ``encode`` / ``decode``: plain-text
    tokenization (the records proxy: the policy tokenizer; the episode driver:
    whitespace words)."""
    ids = list(encode(text))
    if len(ids) <= max_tokens:
        return text, 0
    keep = max(max_tokens - _MARKER_RESERVE, 2)
    head, tail = keep - keep // 2, keep // 2
    dropped = len(ids) - head - tail
    return decode(ids[:head]) + TRUNCATION_MARKER.format(n=dropped) + decode(ids[len(ids) - tail:]), dropped


def _word_cut(text: str, max_tokens: int) -> tuple[str, int]:
    """:func:`truncate_middle` on whitespace words (the episode driver's tokens)."""
    return truncate_middle(text, max_tokens, str.split, " ".join)


def first_task_message(messages: Sequence[Message]) -> Message | None:
    """The task: the first user message before any assistant message (None: none)."""
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            return None
        if role == "user":
            return message
    return None


def rebuild_context(prefix: Sequence[Message], summary: str, steps: Sequence[tuple[str, str]],
                    cfg: CompactionConfig, count: Callable[[str], int], *,
                    task: Message | None = None,
                    cut: Callable[[str, int], tuple[str, int]] = _word_cut) -> tuple[list[Message], int]:
    """Eq. 9: ``s + [task] + u_resume(S) + last k steps``; returns (context, k used).

    k starts at ``cfg.keep_recent_steps`` and is reduced while the rebuilt
    context would trigger compaction again (yeto reading of "reduce k when
    necessary"). ``task`` (kept when ``cfg.keep_task_message``) is cut to
    ``cfg.task_message_max_tokens`` when even k = 0 would trigger with it whole.
    With ``cfg.keep_last_step_max_tokens``, a k of 0 becomes 1: the last step
    with its observation cut to that many tokens, when that fits.
    ``cut(text, max_tokens)``: :func:`truncate_middle` in the caller's tokens.
    """

    resume = {"role": "user", "content": cfg.resume_template.format(summary=summary)}
    head = [*prefix, resume]
    if cfg.keep_task_message and task is not None:
        head = [*prefix, dict(task), resume]
        if should_compact(context_tokens(head, count), cfg):
            text, dropped = cut(str(task["content"]), cfg.task_message_max_tokens)
            if dropped:
                head[len(prefix)] = {**task, "content": text}
    for k in range(min(cfg.keep_recent_steps, len(steps)), 0, -1):
        recent = [m for st in steps[len(steps) - k:] for m in step_messages(st)]
        if not should_compact(context_tokens([*head, *recent], count), cfg):
            return [*head, *recent], k
    if cfg.keep_last_step_max_tokens is not None and cfg.keep_recent_steps and steps:
        action, observation = steps[-1]
        last = step_messages((action, cut(observation, cfg.keep_last_step_max_tokens)[0]))
        if not should_compact(context_tokens([*head, *last], count), cfg):
            return [*head, *last], 1
    return head, 0


@dataclass
class Segment:
    index: int
    prompt: list[Message]  # context the segment starts from
    turns: list[dict[str, Any]] = field(default_factory=list)  # role/content/kind/tokens

    @property
    def trainable_tokens(self) -> int:
        return sum(t["tokens"] for t in self.turns if t["kind"] in ("action", "summary"))


@dataclass
class CompactionEpisode:
    rollout_id: str
    segments: list[Segment]
    reward: float
    compactions: int
    truncated: bool
    summaries_ok: list[bool]
    kept_steps: list[int]  # k used at each rebuild

    def samples(self) -> list[dict[str, Any]]:
        """One training sample per segment (paper: segments optimised individually).

        ``tokens_after`` is N_{>s} (eq. 14, optimised tokens of later segments);
        every segment carries the task reward at its end (shared return).
        """

        n = [s.trainable_tokens for s in self.segments]
        out = []
        for s, seg in enumerate(self.segments):
            out.append({
                "prompt": list(seg.prompt),
                "turns": list(seg.turns),
                "reward": self.reward,
                "metadata": {
                    "rollout_id": self.rollout_id,
                    "segment_index": s,
                    "num_segments": len(self.segments),
                    "segment_tokens": n[s],
                    "tokens_after": sum(n[s + 1:]),
                    # l for length-adaptive lambda: the whole rollout's optimised
                    # length, so every segment shares one lambda (keeps eq. 15).
                    "gae_length": sum(n),
                    "compactions": self.compactions,
                    "truncated": self.truncated,
                },
            })
        return out

    def segment_ids(self) -> list[int]:
        """Per optimised token segment ids over the concatenated rollout.

        Input of the control mode ``--gae-variant cross_segment_whole_rollout``
        (formerly the fork's ``cross_segment``, one sample per rollout; ablation
        only, earlier segments get no terminal reward); training uses
        :meth:`samples` with ``--gae-variant cross_segment_per_sample`` (one
        sample per segment).
        """

        return [s for s, seg in enumerate(self.segments) for _ in range(seg.trainable_tokens)]


def run_episode(policy: Callable[[list[Message]], str], env: Environment,
                prompt: Sequence[Message], cfg: CompactionConfig, *,
                count: Callable[[str], int] = default_count_tokens,
                keep_prefix: int = 1, rollout_id: str | None = None) -> CompactionEpisode:
    """Drive one compacted rollout.

    ``prompt`` is the initial history (system prompt first, then the task);
    ``keep_prefix`` leading messages are the ``s`` kept on rebuild (paper: the
    system prompt; the task itself survives through section 1 of the summary).
    With ``cfg.keep_task_message`` the first user message after them is kept
    verbatim as well (:func:`rebuild_context`).
    """

    prefix = list(prompt[:keep_prefix])
    task = first_task_message(prompt[keep_prefix:])
    ctx: list[Message] = list(prompt)
    seg = Segment(0, list(ctx))
    segments = [seg]
    steps: list[tuple[str, str]] = []
    compactions, truncated = 0, False
    summaries_ok: list[bool] = []
    kept: list[int] = []
    for _ in range(cfg.max_turns):
        action = policy(list(ctx))
        seg.turns.append({"role": "assistant", "content": action, "kind": "action",
                          "tokens": count(action)})
        observation, done = env.step(action)
        seg.turns.append({"role": "user", "content": observation, "kind": "observation",
                          "tokens": count(observation)})
        steps.append((action, observation))
        ctx += step_messages(steps[-1])
        if done:
            break
        used = context_tokens(ctx, count)
        if not should_compact(used, cfg):
            continue
        if compactions >= cfg.max_compactions:
            if used >= cfg.context_budget:
                truncated = True
                break
            continue
        request = {"role": "user", "content": cfg.summary_prompt}
        text = policy([*ctx, request])
        seg.turns.append({"role": "user", "content": cfg.summary_prompt, "kind": "summary_request",
                          "tokens": count(cfg.summary_prompt)})
        seg.turns.append({"role": "assistant", "content": text, "kind": "summary",
                          "tokens": count(text)})
        summary, ok = extract_summary(text)
        summaries_ok.append(ok)
        ctx, k = rebuild_context(prefix, summary, steps, cfg, count, task=task)
        kept.append(k)
        compactions += 1
        seg = Segment(compactions, list(ctx))
        segments.append(seg)
    else:
        truncated = True
    return CompactionEpisode(rollout_id or uuid.uuid4().hex, segments, float(env.reward()),
                             compactions, truncated, summaries_ok, kept)


def validate_compaction_segments(segments: Sequence[Any]) -> None:
    """Check the compaction records of one segmented Miles trajectory
    (moved from the legacy engine's rollout check).

    Every segment carries ``metadata`` with schema version 1, one shared
    non-empty ``compaction_trajectory_id`` and one positive integer
    ``compaction_context_budget``; ``compaction_segment_index`` runs 0..n-1 and
    ``compaction_segment_type`` alternates execution/summary, ending on
    execution; all segments keep one integer (group_index, index) identity.
    Raises ``RuntimeError`` otherwise.
    """
    metadata = [getattr(segment, "metadata", None) for segment in segments]
    trajectory_ids = {
        value.get("compaction_trajectory_id")
        for value in metadata
        if isinstance(value, Mapping)
    }
    indices = [
        value.get("compaction_segment_index")
        if isinstance(value, Mapping)
        else None
        for value in metadata
    ]
    types = [
        value.get("compaction_segment_type")
        if isinstance(value, Mapping)
        else None
        for value in metadata
    ]
    context_budgets = {
        value.get("compaction_context_budget")
        for value in metadata
        if isinstance(value, Mapping)
    }
    if (
        len(trajectory_ids) != 1
        or not all(
            isinstance(value, str) and value for value in trajectory_ids
        )
        or indices != list(range(len(segments)))
        or types != [
            "execution" if value % 2 == 0 else "summary"
            for value in range(len(segments))
        ]
        or types[-1] != "execution"
        or any(
            not isinstance(value, Mapping)
            or value.get("compaction_schema_version") != 1
            for value in metadata
        )
        or len(context_budgets) != 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in context_budgets
        )
    ):
        raise RuntimeError("Miles returned malformed compaction segments")
    identities = {
        (getattr(segment, "group_index", None), getattr(segment, "index", None))
        for segment in segments
    }
    if len(identities) != 1 or any(
        isinstance(value, bool) or not isinstance(value, int)
        for identity in identities
        for value in identity
    ):
        raise RuntimeError("Miles compaction segments changed trajectory identity")


def _optimized_tokens(sample: Any) -> int:
    mask = getattr(sample, "loss_mask", None) or []
    return int(sum(1 for value in mask if value))


def refresh_segment_counts(samples: Any) -> int:
    """Recompute ``segment_tokens``/``tokens_after``/``gae_length`` from the final loss masks.

    Segment samples (``metadata.segment_index``) count optimised tokens when
    they are assembled; later hooks still set loss_mask 1 -> 0 (placeholder-logprob
    stop tokens, reasoning tokens) before the
    backend converts the samples. N_{>s} and l are optimised-token counts, so
    they are refreshed after those hooks (otherwise tokens_after
    comes out 1-17 tokens high). Returns the number of segment samples
    whose counts changed.
    """
    rollouts: dict[Any, list[Any]] = {}
    for sample in samples:
        meta = getattr(sample, "metadata", None)
        if not isinstance(meta, dict) or not isinstance(meta.get("segment_index"), int):
            continue
        key = (getattr(sample, "rollout_id", None), getattr(sample, "group_index", None),
               getattr(sample, "index", None))
        rollouts.setdefault(key, []).append(sample)
    changed = 0
    for segments in rollouts.values():
        segments.sort(key=lambda s: s.metadata["segment_index"])
        tokens = [_optimized_tokens(s) for s in segments]
        for index, sample in enumerate(segments):
            new = {"segment_tokens": tokens[index], "tokens_after": sum(tokens[index + 1:]),
                   "gae_length": sum(tokens)}
            if any(sample.metadata.get(k) != v for k, v in new.items()):
                changed += 1
            sample.metadata.update(new)
    return changed


# --- the sampler spec's ``agentic.compaction`` section -------------------------
# Every CompactionConfig field is a spec key except ``max_turns``: the episode
# driver's evaluation cap, which the records proxy does not read (an accepted key
# must reach its consumer). CompactionConfig checks the summary-turn fields
# (``summary_max_tokens``, ``summary_prompt``, ``resume_template``) and the
# rebuild fields (``keep_task_message``, ``keep_last_step_max_tokens``: null is off).
SPEC_KEYS: tuple[str, ...] = tuple(f.name for f in fields(CompactionConfig) if f.name != "max_turns")
# Integer keys and their least value (``summary_max_tokens``: CompactionConfig checks it).
_INT_MINIMUM = {"context_budget": 1, "t_comp": 1, "keep_recent_steps": 0, "max_compactions": 0,
                "task_message_max_tokens": MIN_CUT_TOKENS}


def _spec_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"agentic.compaction.{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"agentic.compaction.{name} must be >= {minimum}, got {value}")
    return value


def spec_max_model_len(spec: Mapping[str, Any]) -> int | None:
    """The engine context length a sampler spec declares (``engine.server.max_model_len``,
    or ``engine.kwargs.max_model_len`` of an adapter without a server); None if unset."""
    engine = spec.get("engine") if isinstance(spec.get("engine"), Mapping) else {}
    for part in ("server", "kwargs"):
        section = engine.get(part)
        if isinstance(section, Mapping) and section.get("max_model_len") is not None:
            return int(section["max_model_len"])
    return None


def _check_summary_room(cfg: CompactionConfig, turn_max_tokens: int, max_model_len: int | None) -> None:
    """The summary turn must fit: in ``t_comp`` (:meth:`CompactionConfig.check_summary_room`)
    and, when the engine context is known, within ``max_model_len``
    (:meth:`CompactionConfig.check_summary_headroom`)."""
    try:
        cfg.check_summary_room(turn_max_tokens)
        if max_model_len is not None:
            cfg.check_summary_headroom(max_model_len, turn_max_tokens)
    except ValueError as exc:
        raise ValueError(f"agentic.compaction: {exc}") from None


def _check_agent(section: Mapping[str, Any]) -> None:
    """The proxy compacts on the history it records; an agent that compacts its own
    history rewrites it and breaks the token chain. Only agents whose own compaction
    yeto switches off may run (:data:`yeto.rl.harness.harbor.pi_config.PROXY_COMPACTION_AGENTS`);
    the default agent of the spec is checked here, each task's agent when its trial
    is built."""
    from yeto.rl.harness.harbor.pi_config import PROXY_COMPACTION_AGENTS  # noqa: PLC0415 - harbor package

    defaults = section.get("task_defaults") if isinstance(section.get("task_defaults"), Mapping) else {}
    agent = defaults.get("agent", defaults.get("agent_name"))
    if agent is not None and str(agent) not in PROXY_COMPACTION_AGENTS:
        raise ValueError(f"agentic.compaction: agentic.task_defaults.agent {agent!r} may compact its own history; "
                         f"the records proxy's compaction needs an agent whose own compaction is off "
                         f"({', '.join(sorted(PROXY_COMPACTION_AGENTS))})")


def segment_width(spec: Mapping[str, Any], cfg: CompactionConfig, *,
                  max_model_len: int | None = None) -> tuple[int, str | None]:
    """``(tokens, warning)``: how wide a compaction segment's sample can be, which the
    sampler's ``agentic.max_response_len`` and the trainer's ``--seq-len`` must hold.

    The trigger is checked before a model call: a step that did not trigger adds its
    reply and then tool output of any length, and the summary turn follows, so a
    segment's stream can grow to the engine context (``max_model_len``), past
    ``context_budget``. A non-last segment cut at ``max_response_len`` loses its
    summary and the whole trial is dropped. Without a known engine context the width
    is ``context_budget`` and ``warning`` says it is not enough."""
    limit = spec_max_model_len(spec) if max_model_len is None else int(max_model_len)
    if limit is not None:
        return limit, None
    return cfg.context_budget, (
        f"agentic.compaction: the spec gives no engine max_model_len, so the row widths are checked against "
        f"context_budget {cfg.context_budget}; a segment can grow past it to the engine context (a large tool "
        f"output after the last step below the trigger, then the summary), and a non-last segment cut at "
        f"agentic.max_response_len drops the whole trial: set engine.server.max_model_len and keep "
        f"agentic.max_response_len and the trainer's --seq-len at least that")


def config_from_spec(spec: Mapping[str, Any], *, max_model_len: int | None = None) -> CompactionConfig | None:
    """``agentic.compaction`` of a sampler spec as a :class:`CompactionConfig`, or None
    (no section, or ``null``: no compaction).

    Only ``context_budget`` is required; the other keys default to the paper's
    values (:data:`T_COMP`, :data:`KEEP_RECENT_STEPS`, :data:`MAX_COMPACTIONS`).
    Refused (ValueError), so that ``pool --dry-run`` and the sampler's start stop it:

    * unknown keys (:data:`SPEC_KEYS`), non-integers, ``t_comp < 1``,
      ``keep_recent_steps`` / ``max_compactions`` below 0, ``context_budget <= t_comp``;
    * ``context_budget`` above the engine context length (``max_model_len``, else the
      spec's :func:`spec_max_model_len`): the engine refuses a prompt the proxy still
      lets grow;
    * a ``t_comp`` that cannot hold the summary turn, and a budget whose summary
      turn can overrun the engine context (:func:`_check_summary_room`, with
      ``agentic.turn_max_tokens``);
    * ``agentic.max_response_len`` below the engine context length (:func:`segment_width`;
      ``context_budget`` when the spec gives none): a segment's response is cut at
      that width, and the summary at the end of a segment is what is cut;
    * a default agent that compacts its own history (:func:`_check_agent`)."""
    agentic = spec.get("agentic")
    raw = agentic.get("compaction") if isinstance(agentic, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"agentic.compaction must be an object or null, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(SPEC_KEYS))
    if unknown:
        raise ValueError(f"agentic.compaction: unknown key(s) {unknown} (known: {list(SPEC_KEYS)})")
    if "context_budget" not in raw:
        raise ValueError("agentic.compaction.context_budget is required (the context size C that triggers "
                         "a compaction)")
    values = {key: _spec_int(key, raw[key], minimum=least) for key, least in _INT_MINIMUM.items() if key in raw}
    values.update({key: value for key, value in raw.items() if key not in _INT_MINIMUM})
    budget, t_comp = values["context_budget"], values.get("t_comp", T_COMP)
    if budget <= t_comp:
        raise ValueError(f"agentic.compaction.context_budget {budget} must exceed t_comp {t_comp}")
    limit = spec_max_model_len(spec) if max_model_len is None else int(max_model_len)
    if limit is not None and budget > limit:
        raise ValueError(f"agentic.compaction.context_budget {budget} exceeds the engine context length "
                         f"max_model_len {limit}: the engine refuses a prompt before the proxy compacts it")
    try:
        cfg = CompactionConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic.compaction: {exc}") from None
    for key in ("turn_max_tokens", "max_response_len"):
        if agentic.get(key) is None:
            raise ValueError(f"agentic.compaction needs agentic.{key}")
    _check_summary_room(cfg, int(agentic["turn_max_tokens"]), limit)
    width, _warning = segment_width(spec, cfg, max_model_len=limit)
    if int(agentic["max_response_len"]) < width:
        what = f"the engine context length max_model_len {width}" if limit is not None else \
            f"agentic.compaction.context_budget {width}"
        raise ValueError(f"agentic.max_response_len {int(agentic['max_response_len'])} < {what}: a compaction "
                         f"segment can grow to the engine context (the trigger is checked before a model call, "
                         f"so a large tool output can follow the last step below it, then the summary), and a "
                         f"segment cut at max_response_len loses the summary that ends it and drops the trial; "
                         f"set max_response_len >= {width}")
    _check_agent(agentic)
    return cfg


# The summary_max_tokens the dry runs suggest when keep_task_message is on (summary_length_note).
SUGGESTED_SUMMARY_MAX_TOKENS = 2048


def summary_length_note(cfg: CompactionConfig, turn_max_tokens: int,
                        max_model_len: int | None = None) -> str | None:
    """A dry-run note (None: nothing to say): with ``keep_task_message`` on, the
    summary request asks for identifiers verbatim and summaries run longer, so a
    summary turn below :data:`SUGGESTED_SUMMARY_MAX_TOKENS` is often cut. Only
    when the spec's room allows that many (``t_comp`` holds it,
    :meth:`CompactionConfig.check_summary_room`, and the engine context when known,
    :meth:`CompactionConfig.check_summary_headroom`). Advice only: the defaults do
    not change."""
    tokens = cfg.summary_tokens(turn_max_tokens)
    if not cfg.keep_task_message or tokens >= SUGGESTED_SUMMARY_MAX_TOKENS:
        return None
    try:
        bigger = replace(cfg, summary_max_tokens=SUGGESTED_SUMMARY_MAX_TOKENS)
        if max_model_len is not None:
            bigger.check_summary_headroom(max_model_len, turn_max_tokens)
    except ValueError:
        return None
    return (f"agentic.compaction: the summary turn's max_tokens is {tokens}; with keep_task_message on the "
            f"summary keeps identifiers verbatim and small models often reach that cap (a cut summary is counted "
            f"as summary_truncated). The spec has room for summary_max_tokens {SUGGESTED_SUMMARY_MAX_TOKENS}: "
            f"consider setting it")


def spec_summary_length_note(spec: Mapping[str, Any], cfg: CompactionConfig) -> str | None:
    """:func:`summary_length_note` for a sampler spec (``agentic.turn_max_tokens``, the
    spec's engine context); None without a turn cap."""
    agentic = spec.get("agentic") if isinstance(spec.get("agentic"), Mapping) else {}
    if agentic.get("turn_max_tokens") is None:
        return None
    return summary_length_note(cfg, int(agentic["turn_max_tokens"]), spec_max_model_len(spec))


def describe(cfg: CompactionConfig) -> str:
    """One line of the dry runs: the effective values of the section."""
    line = (f"on: context_budget {cfg.context_budget}, t_comp {cfg.t_comp} (compacts once fewer than "
            f"{cfg.t_comp} tokens remain), keep_recent_steps {cfg.keep_recent_steps}, max_compactions "
            f"{cfg.max_compactions} (at most {cfg.max_compactions + 1} segments per trial)")
    if cfg.summary_max_tokens is not None:
        line += f", summary_max_tokens {cfg.summary_max_tokens}"
    else:
        line += f", summary max_tokens: the turn's, at most {DEFAULT_SUMMARY_MAX_TOKENS}"
    if cfg.keep_task_message:
        line += (f", keep_task_message on (the task message is kept verbatim in a rebuilt context, cut to "
                 f"task_message_max_tokens {cfg.task_message_max_tokens} only when it would keep it over the trigger)")
    else:
        line += ", keep_task_message off (a rebuilt context keeps the task only through the summary)"
    if cfg.keep_last_step_max_tokens is not None:
        line += (f", keep_last_step_max_tokens {cfg.keep_last_step_max_tokens} (when no whole step fits, the "
                 f"last step is kept with its tool output cut to that many tokens)")
    else:
        line += ", keep_last_step_max_tokens off (when no whole step fits, no step is kept)"
    if cfg.max_compactions == 0:
        line += ("; max_compactions 0: the proxy never compacts, but still ends a trial at context_budget "
                 "(remove agentic.compaction to turn it off)")
    return line
