"""CompactionRL rollout side.

Pure-Python episode driver for compacted agent rollouts, after CompactionRL
(arXiv 2607.05378v1, sec. 4.1 eq. 6-9, sec. 5.1):

* trigger (eq. 7): compact when the remaining context ``C - |h_t| < T_comp``
  (``T_comp`` = 10,240);
* the *same* policy writes the summary ``S_t ~ pi(. | h_t + q_sum)`` (eq. 8),
  with an ``<analysis>`` scratch part and a ``<summary>`` part;
* rebuilt context (eq. 9): ``s + u_resume(S_t) + last k steps`` (k = 2, reduced
  when the rebuilt context would trigger again); a step = (action, observation)
  is atomic (eq. 6);
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
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

T_COMP = 10_240
KEEP_RECENT_STEPS = 2
MAX_COMPACTIONS = 3

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
    "First think inside <analysis></analysis>: go through the history in order and "
    "check what must not be lost.\n"
    "Then write the summary inside <summary></summary> with exactly these numbered "
    "sections:\n"
    + "".join(f"{i}. {name}\n" for i, name in enumerate(SUMMARY_SECTIONS, 1))
    + "Do not call any tool in this turn."
)

RESUME_TEMPLATE = (
    "This session continues from an earlier one whose context was compacted. "
    "Summary of the earlier interaction:\n\n{summary}\n\n"
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
    ``resume_template`` has one ``{summary}`` field."""

    context_budget: int  # C (paper: 64k GLM-4.7-Flash, 80k GLM-4.5-Air-SFT)
    t_comp: int = T_COMP
    keep_recent_steps: int = KEEP_RECENT_STEPS
    max_compactions: int = MAX_COMPACTIONS
    max_turns: int = 250  # evaluation turn cap (sec. 5.1); training value not given
    summary_max_tokens: int | None = None
    summary_prompt: str = SUMMARY_PROMPT
    resume_template: str = RESUME_TEMPLATE

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


def step_messages(step: tuple[str, str]) -> list[Message]:
    action, observation = step
    return [{"role": "assistant", "content": action}, {"role": "user", "content": observation}]


def rebuild_context(prefix: Sequence[Message], summary: str, steps: Sequence[tuple[str, str]],
                    cfg: CompactionConfig, count: Callable[[str], int]) -> tuple[list[Message], int]:
    """Eq. 9: ``s + u_resume(S) + last k steps``; returns (context, k used).

    k starts at ``cfg.keep_recent_steps`` and is reduced while the rebuilt
    context would trigger compaction again (yeto reading of "reduce k when
    necessary").
    """

    resume = {"role": "user", "content": cfg.resume_template.format(summary=summary)}
    for k in range(min(cfg.keep_recent_steps, len(steps)), -1, -1):
        recent = [m for st in steps[len(steps) - k:] for m in step_messages(st)] if k else []
        ctx = [*prefix, resume, *recent]
        if not should_compact(context_tokens(ctx, count), cfg) or k == 0:
            return ctx, k
    raise AssertionError("unreachable")


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
    """

    prefix = list(prompt[:keep_prefix])
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
        ctx, k = rebuild_context(prefix, summary, steps, cfg, count)
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
# must reach its consumer). The summary-turn fields (``summary_max_tokens``,
# ``summary_prompt``, ``resume_template``) are accepted once CompactionConfig has
# them; CompactionConfig checks their values.
SPEC_KEYS: tuple[str, ...] = tuple(f.name for f in fields(CompactionConfig) if f.name != "max_turns")
# Integer keys and their least value (``summary_max_tokens``: CompactionConfig checks it).
_INT_MINIMUM = {"context_budget": 1, "t_comp": 1, "keep_recent_steps": 0, "max_compactions": 0}


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


def _check_summary_room(cfg: CompactionConfig, turn_max_tokens: int) -> None:
    """``t_comp`` must hold the summary turn: its ``max_tokens`` plus the summary request.
    CompactionConfig owns the rule (``check_summary_room``, with the configurable
    summary prompt and ``summary_max_tokens``). Until that method is merged the
    summary turn is a plain turn (``turn_max_tokens``) with :data:`SUMMARY_PROMPT`,
    and the same bound is applied here: one token per UTF-8 byte of the prompt plus
    64 for the template's wrapper."""
    check = getattr(cfg, "check_summary_room", None)
    try:
        if check is not None:
            check(turn_max_tokens)
            return
        need = turn_max_tokens + len(SUMMARY_PROMPT.encode("utf-8")) + 64
        if need > cfg.t_comp:
            raise ValueError(f"t_comp {cfg.t_comp} cannot hold the summary turn: max_tokens {turn_max_tokens} "
                             f"(agentic.turn_max_tokens) + up to {need - turn_max_tokens} tokens of summary "
                             f"request = {need}; raise t_comp or lower agentic.turn_max_tokens")
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
    * a ``t_comp`` that cannot hold the summary turn (:func:`_check_summary_room`,
      with ``agentic.turn_max_tokens``), and, where CompactionConfig has
      ``check_summary_headroom``, a budget whose summary turn can overrun the
      engine context;
    * ``agentic.max_response_len`` below ``context_budget``: a segment's response is
      cut at that width, and the summary at the end of a segment is what is cut;
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
    _check_summary_room(cfg, int(agentic["turn_max_tokens"]))
    headroom = getattr(cfg, "check_summary_headroom", None)  # the summary turn within the engine context
    if headroom is not None and limit is not None:
        try:
            headroom(limit, int(agentic["turn_max_tokens"]))
        except ValueError as exc:
            raise ValueError(f"agentic.compaction: {exc}") from None
    if int(agentic["max_response_len"]) < budget:
        raise ValueError(f"agentic.max_response_len {int(agentic['max_response_len'])} < agentic.compaction."
                         f"context_budget {budget}: a segment's response is cut at max_response_len, and the "
                         f"summary that ends the segment is the part cut; set max_response_len >= context_budget")
    _check_agent(agentic)
    return cfg


def describe(cfg: CompactionConfig) -> str:
    """One line of the dry runs: the effective values of the section."""
    line = (f"on: context_budget {cfg.context_budget}, t_comp {cfg.t_comp} (compacts once fewer than "
            f"{cfg.t_comp} tokens remain), keep_recent_steps {cfg.keep_recent_steps}, max_compactions "
            f"{cfg.max_compactions} (at most {cfg.max_compactions + 1} segments per trial)")
    summary_max_tokens = getattr(cfg, "summary_max_tokens", None)
    if summary_max_tokens is not None:
        line += f", summary_max_tokens {summary_max_tokens}"
    if cfg.max_compactions == 0:
        line += "; max_compactions 0: the proxy never compacts"
    return line
