"""S17 C9: Codex end reason split + per-turn lengths reach the trajectory event."""

from __future__ import annotations

import asyncio

import pytest

from yeto.rl.engine import timeline
from yeto.rl.engine.miles_adapter.rollout_meta_hook import trajectory_diagnostics
from yeto.rl.harness.codex import agent as legacy
from yeto.rl.harness.codex import codex_harness_agent as H
from yeto.rl.harness.codex import codex_openenv_agent_function as F


def _driver(exc=None, submit_calls=0):
    metrics = legacy.AgentMetrics(submit_calls=submit_calls)
    d = H._AppServerDriver.__new__(H._AppServerDriver)
    d._metrics = metrics

    async def proto():
        if exc is not None:
            raise exc
        return "completed"

    d._drive_protocol = proto
    return d, metrics


@pytest.mark.parametrize("exc,status,kind", [
    (H.CodexModelFailure("DSV4 must return exactly one tool call"), "max_turns", "protocol_error"),
    (H.CodexTurnLimit("Codex reached the signed turn budget"), "max_turns", "turn_limit"),
    (H.CodexSequenceLimit("Miles returned a truncated sample"), "max_seq_len", "response_truncated"),
    (H.CodexSequenceLimit("Miles sequence limit reached"), "max_seq_len", "context_limit"),
    (None, "completed", "completed_no_submit"),
])
def test_drive_splits_end_kind(exc, status, kind):
    d, m = _driver(exc)
    assert asyncio.run(d.drive()) == status  # signed status unchanged
    assert m.end_kind == kind and kind in H.END_KINDS


def test_drive_submit():
    d, m = _driver(None, submit_calls=1)
    assert asyncio.run(d.drive()) == "completed" and m.end_kind == "submit"


def test_end_kind_for_rejects_infra_errors():
    with pytest.raises(TypeError):
        H.end_kind_for(H.CodexHarnessError("x"))


def test_turn_lengths_recorded_and_capped():
    m = legacy.AgentMetrics()
    choice = {"finish_reason": "tool_calls", "message": {"content": "", "reasoning_content": "r"}}
    H._note_last_completion(m, choice, {"usage": {"completion_tokens": 120, "total_tokens": 900}})
    H._note_last_completion(m, choice, {})
    H._note_turn_tool_output(m, "héllo")
    assert m.turn_completion_tokens == [120, None]
    assert m.turn_context_tokens == [900, None]
    assert m.turn_tool_output_bytes == [6]
    for _ in range(H.TURN_LENGTHS_MAX + 5):
        H._note_turn_tool_output(m, "x")
    assert len(m.turn_tool_output_bytes) == H.TURN_LENGTHS_MAX
    d = F._metrics_dict(m)
    assert d["turn_completion_tokens"] == [120, None] and "end_kind" not in d
    m.end_kind = "protocol_error"
    assert F._metrics_dict(m)["end_kind"] == "protocol_error"


def test_event_carries_fields_and_validates():
    meta = {"exit_status": "max_turns",
            "agent_metrics": {"turns": 1, "parse_failures": 1, "end_kind": "protocol_error",
                              "turn_completion_tokens": [4100, True, None] + [1] * 300,
                              "turn_context_tokens": [5000], "turn_tool_output_bytes": []}}
    out = trajectory_diagnostics(meta)
    assert out["exit_status"] == "max_turns" and out["end_kind"] == "protocol_error"
    assert out["turn_completion_tokens"][:3] == [4100, None, None]
    assert len(out["turn_completion_tokens"]) == 256
    record = {"rollout_id": 0, "policy_version": 0, "sample_index": 0, "group_index": 0, "task_id": "t",
              "trajectory_id": "x", "reward": 0.0, "success": False, "aborted": False, **out}
    assert timeline.validate_trajectory_reward(record) == []
    assert timeline.validate_trajectory_reward({**record, "end_kind": 3})
