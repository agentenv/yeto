"""Fields the async GRPO dashboard reads from the senders (observation only):
the engine sampler's round summary and the buffer trainer's batch summary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import rollout_meta_hook as rmh
from yeto.rl.engine.engine_sampler import round_summary, zero_variance


def _group(rewards, statuses=None, length=10):
    statuses = statuses or ["completed"] * len(rewards)
    return [{"reward": r, "status": s, "response_length": length + i} for i, (r, s) in enumerate(zip(rewards, statuses))]


def test_round_summary_reward_length_truncation_zero_variance():
    scored = [(None, _group([1.0, 0.0, 1.0, 0.0], ["completed", "truncated", "completed", "completed"])),
              (None, _group([1.0, 1.0, 1.0, 1.0]))]
    s = round_summary(scored)
    assert s["reward_mean"] == pytest.approx(0.75)
    assert s["reward_std"] == pytest.approx(0.4330127)
    assert (s["reward_p10"], s["reward_p50"], s["reward_p90"]) == (0.0, 1.0, 1.0)
    assert s["resp_len_mean"] == pytest.approx(11.5) and s["resp_len_p95"] == pytest.approx(13.0)
    assert s["truncated"] == 1
    assert s["zero_variance_seen"] == 1 == sum(zero_variance(g) for _, g in scored)


def test_round_summary_empty_round_adds_nothing():
    assert round_summary([]) == {}


def _sample(i, reward, status="completed", length=10):
    return SimpleNamespace(index=i, group_index=0, rollout_id=3, status=SimpleNamespace(value=status),
                           remove_sample=False, metadata=None, reward=reward, response_length=length,
                           weight_versions=None)


def test_buffer_reader_round_gets_batch_summary(monkeypatch):
    monkeypatch.delenv(rmh.BATCH_SUMMARY_ENV, raising=False)
    monkeypatch.setattr(rmh, "current_round_id", lambda *a, **k: 3)
    data = [[_sample(0, 1.0), _sample(1, 0.0, "truncated", 30)]]
    plain = SimpleNamespace()
    rmh.record_trained_groups(plain, data)
    assert "batch_summary" not in rmh.build_metadata(plain, data)  # default key set unchanged
    reader = SimpleNamespace()
    rmh.set_proc_state(reader, rmh.BUFFER_READER_ATTR, True)
    rmh.record_trained_groups(reader, data)
    meta = rmh.build_metadata(reader, data)
    assert meta["batch_summary"]["truncated_frac"] == 0.5
    assert meta["batch_summary"]["reward_p90"] == pytest.approx(0.9)
    # the opt-in per-trajectory records stay opt-in
    assert rmh.TRAJECTORY_REWARDS_KEY not in meta
    assert rmh.buffer_summary_enabled(reader) and not rmh.batch_summary_enabled(reader)
