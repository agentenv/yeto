"""CompactionRL rollout side (change ``rl-algo-critic-family`` task 9.2, design D8).

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
:data:`RESUME_TEMPLATE` are yeto drafts (design D8 asks for a 9-section
template; the paper's sec. 4.1 lists what must be kept: original goal, completed
actions, important observations, unresolved errors, current state, plausible
next steps) and need user confirmation.

Integration (task 9.1 gap): the driver takes a ``policy(messages) -> str`` and
an ``env`` with ``step(action) -> (observation, done)`` / ``reward()``; wiring it
into the Codex / Terminal-Bench harness or Miles session server is not done.
No torch / miles imports.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CompactionConfig:
    context_budget: int  # C (paper: 64k GLM-4.7-Flash, 80k GLM-4.5-Air-SFT)
    t_comp: int = T_COMP
    keep_recent_steps: int = KEEP_RECENT_STEPS
    max_compactions: int = MAX_COMPACTIONS
    max_turns: int = 250  # evaluation turn cap (sec. 5.1); training value not given

    def __post_init__(self) -> None:
        if self.context_budget <= self.t_comp:
            raise ValueError("context_budget must exceed t_comp")
        if self.keep_recent_steps < 0 or self.max_compactions < 0 or self.max_turns < 1:
            raise ValueError("keep_recent_steps/max_compactions must be >= 0, max_turns >= 1")


def context_tokens(messages: Sequence[Message], count: Callable[[str], int]) -> int:
    return sum(count(m["content"]) for m in messages)


def should_compact(used_tokens: int, cfg: CompactionConfig) -> bool:
    """Eq. 7: ``C - |h_t| < T_comp``."""

    return cfg.context_budget - used_tokens < cfg.t_comp


def extract_summary(text: str) -> tuple[str, bool]:
    """``(summary, ok)``: the ``<summary>`` body, ``<analysis>`` dropped.

    Without a closed ``<summary>`` tag the text minus any analysis block is
    used and ``ok`` is False (recorded in the segment metadata).
    """

    m = re.search(r"<summary>(.*?)</summary>", text, flags=re.S)
    if m:
        return m.group(1).strip(), True
    stripped = re.sub(r"<analysis>.*?(</analysis>|$)", "", text, flags=re.S)
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

    resume = {"role": "user", "content": RESUME_TEMPLATE.format(summary=summary)}
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
                    "compactions": self.compactions,
                    "truncated": self.truncated,
                },
            })
        return out

    def segment_ids(self) -> list[int]:
        """Per optimised token segment ids over the concatenated rollout.

        The layout fork ce96fc060 reads from ``sample.metadata['segment_ids']``
        (one sample per rollout); see progress.md "S13 9.x" for why per-segment
        samples are needed instead.
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
        request = {"role": "user", "content": SUMMARY_PROMPT}
        text = policy([*ctx, request])
        seg.turns.append({"role": "user", "content": SUMMARY_PROMPT, "kind": "summary_request",
                          "tokens": count(SUMMARY_PROMPT)})
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
