"""rl-algo-critic-family 9.2: CompactionRL rollout driver with a fake model (CPU)."""

from __future__ import annotations

import pytest

from yeto.rl import compaction as cp
from yeto.rl.compaction import CompactionConfig, run_episode


class FakeEnv:
    def __init__(self, done_after: int, obs_tokens: int, reward: float = 1.0):
        self.done_after, self.obs_tokens, self._reward, self.n = done_after, obs_tokens, reward, 0

    def step(self, action):
        self.n += 1
        return " ".join([f"o{self.n}"] * self.obs_tokens), self.n >= self.done_after

    def reward(self):
        return self._reward


class FakePolicy:
    """Actions of ``act_tokens`` words; a summary when the last message is q_sum."""

    def __init__(self, act_tokens: int, summary_ok: bool = True):
        self.act_tokens, self.summary_ok, self.calls, self.n = act_tokens, summary_ok, [], 0

    def __call__(self, messages):
        self.calls.append(messages)
        if messages[-1]["content"] == cp.SUMMARY_PROMPT:
            body = f"S{sum(1 for c in self.calls if c[-1]['content'] == cp.SUMMARY_PROMPT)}"
            if self.summary_ok:
                return f"<analysis>scratch notes</analysis><summary>{body}</summary>"
            return f"<analysis>scratch notes</analysis>{body}"
        self.n += 1
        return " ".join([f"a{self.n}"] * self.act_tokens)


PROMPT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task words here"}]


def cfg(**kw):
    return CompactionConfig(**{"context_budget": 100, "t_comp": 40, **kw})


def test_trigger_is_remaining_budget_below_t_comp():
    c = CompactionConfig(context_budget=65536)
    assert c.t_comp == 10240 and c.keep_recent_steps == 2 and c.max_compactions == 3
    assert not cp.should_compact(65536 - 10240, c)
    assert cp.should_compact(65536 - 10239, c)


def test_no_compaction_when_context_stays_small():
    ep = run_episode(FakePolicy(5), FakeEnv(3, 5), PROMPT, cfg(), rollout_id="r")
    assert ep.compactions == 0 and len(ep.segments) == 1 and not ep.truncated
    assert ep.segment_ids() == [0] * 15
    (s,) = ep.samples()
    assert s["metadata"]["tokens_after"] == 0 and s["reward"] == 1.0


def test_compaction_segments_numbering_and_shared_reward():
    # each step adds 20 tokens; prompt 4 -> compaction once used > 60
    ep = run_episode(FakePolicy(10), FakeEnv(8, 10, reward=0.5), PROMPT, cfg(), rollout_id="r")
    assert ep.compactions >= 2
    samples = ep.samples()
    assert [s["metadata"]["segment_index"] for s in samples] == list(range(len(samples)))
    assert {s["metadata"]["num_segments"] for s in samples} == {len(samples)}
    assert {s["metadata"]["rollout_id"] for s in samples} == {"r"}
    assert {s["reward"] for s in samples} == {0.5}
    n = [s["metadata"]["segment_tokens"] for s in samples]
    assert [s["metadata"]["tokens_after"] for s in samples] == [sum(n[i + 1:]) for i in range(len(n))]
    ids = ep.segment_ids()
    assert ids == sorted(ids) and [ids.count(i) for i in range(len(n))] == n
    # the summary is trainable and closes every segment but the last
    for seg in ep.segments[:-1]:
        assert seg.turns[-1]["kind"] == "summary"
        assert seg.turns[-2]["kind"] == "summary_request"
    assert all(t["kind"] != "summary" for t in ep.segments[-1].turns)


def test_rebuilt_context_is_system_resume_and_last_k_steps():
    policy = FakePolicy(5)
    ep = run_episode(policy, FakeEnv(20, 5), PROMPT,
                     CompactionConfig(context_budget=200, t_comp=120), rollout_id="r")
    assert ep.compactions >= 1 and ep.kept_steps[0] == 2
    seg1 = ep.segments[1].prompt
    assert seg1[0] == PROMPT[0]  # s = system prompt only
    assert seg1[1]["role"] == "user"
    assert seg1[1]["content"] == cp.RESUME_TEMPLATE.format(summary="S1")
    assert "scratch notes" not in seg1[1]["content"]  # <analysis> dropped
    recent = seg1[2:]
    assert len(recent) == 4 and [m["role"] for m in recent] == ["assistant", "user"] * 2
    prev = ep.segments[0].turns
    actions = [t["content"] for t in prev if t["kind"] == "action"]
    assert [recent[0]["content"], recent[2]["content"]] == actions[-2:]
    # the summary request saw the full history plus q_sum
    req = next(c for c in policy.calls if c[-1]["content"] == cp.SUMMARY_PROMPT)
    assert req[:2] == PROMPT


def test_k_is_reduced_when_recent_steps_do_not_fit():
    ep = run_episode(FakePolicy(15), FakeEnv(20, 15), PROMPT,
                     CompactionConfig(context_budget=100, t_comp=50), rollout_id="r")
    assert ep.compactions >= 1
    assert ep.kept_steps[0] < 2
    for seg in ep.segments[1:]:
        assert not cp.should_compact(cp.context_tokens(seg.prompt, cp.default_count_tokens),
                                     CompactionConfig(context_budget=100, t_comp=50))


def test_at_most_three_compactions_then_truncated_when_full():
    ep = run_episode(FakePolicy(10), FakeEnv(1000, 10), PROMPT, cfg(), rollout_id="r")
    assert ep.compactions == 3 and len(ep.segments) == 4
    assert ep.truncated
    assert ep.segment_ids()[-1] == 3


def test_max_compactions_is_configurable_and_zero_disables():
    ep = run_episode(FakePolicy(10), FakeEnv(1000, 10), PROMPT, cfg(max_compactions=0))
    assert ep.compactions == 0 and ep.truncated and len(ep.segments) == 1


def test_missing_summary_tag_is_recorded():
    ep = run_episode(FakePolicy(10, summary_ok=False), FakeEnv(8, 10), PROMPT, cfg())
    assert ep.summaries_ok and not any(ep.summaries_ok)
    assert "scratch" not in ep.segments[1].prompt[1]["content"]


def test_summary_template_has_nine_sections_and_tags():
    assert len(cp.SUMMARY_SECTIONS) == 9
    assert "<analysis>" in cp.SUMMARY_PROMPT and "<summary>" in cp.SUMMARY_PROMPT
    for i, name in enumerate(cp.SUMMARY_SECTIONS, 1):
        assert f"{i}. {name}" in cp.SUMMARY_PROMPT


def test_config_validation():
    with pytest.raises(ValueError):
        CompactionConfig(context_budget=10240)
