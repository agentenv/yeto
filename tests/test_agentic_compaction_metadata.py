"""``metadata.compaction`` fields for the round metrics: ``summary_tokens`` and
``retry_replays`` (CPU; the pieces and fakes of test_agentic_compaction_segments),
and the trainer's group check (``validate_compaction_group``) on real segment samples."""

from __future__ import annotations

import copy

import pytest
from test_agentic_compaction_segments import (SUMMARY_END, Agent, _by_trial, _cfg, _engine, _proxy, _round,
                                              _tool_text, glm)  # noqa: F401 - glm is a fixture
from test_records_proxy import echo, responses_body, tool_output, user_item

from yeto.rl.agentic import TrialVerdict
from yeto.rl.contracts import validate_compaction_group
from yeto.rl.engine.buffer_payload import check_trials
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

pytest.importorskip("aiohttp")


def _summary_lengths(engine, template) -> list[int]:
    """Tokens of every summary completion the engine sampled, in call order."""
    return [len(c.token_ids) for prompt, c in engine.completions
            if SUMMARY_END in template.decode(list(prompt[-30:]))]


def test_each_segment_reports_its_summary_tokens_and_the_trials_retry_replays(tmp_path, glm):
    engine = _engine(glm)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, _cfg()), allow_compaction=True)
    samples, _ = _round(tmp_path, recorder, engine, Agent(5), n=1)
    trial = _by_trial(samples)[0]
    metas = [s["metadata"]["compaction"] for s in trial]
    assert len(metas) == 4
    assert [m["summary_tokens"] for m in metas] == [*_summary_lengths(engine, glm), 0]  # 0 on the last
    assert all(m["summary_tokens"] > 0 for m in metas[:-1])
    assert all(m["summary_tokens"] <= m["segment_tokens"] for m in metas)  # summary tokens are trained
    assert {m["retry_replays"] for m in metas} == {0}  # replay is on with compaction; nothing was resent
    (checked,) = validate_compaction_group(samples)  # the trainer's check takes the real samples
    assert checked.num_segments == 4 and check_trials(samples) == [list(checked.positions)]


class ResendingAgent(Agent):
    """:class:`Agent` that sends its second request twice (a client retry after a lost reply)."""

    async def run(self, task, endpoint):
        import aiohttp

        self.calls += 1
        items = [user_item("do the task")]
        async with aiohttp.ClientSession() as http:
            for turn in range(self.steps):
                for _ in range(2 if turn == 1 else 1):
                    body = responses_body(copy.deepcopy(items))
                    async with http.post(f"{endpoint}/responses", json=body) as reply:
                        assert reply.status == 200, await reply.text()
                        response = await reply.json()
                items += echo(response["output"])
                items += [tool_output(o["call_id"], _tool_text(turn, turn == self.steps - 2))
                          for o in response["output"] if o["type"] == "function_call"]
        return TrialVerdict(rewards={"reward": 1.0})


def test_a_resent_request_is_counted_on_every_segment_of_its_trial(tmp_path, glm):
    engine = _engine(glm)
    recorder = ProxyTokenRecorder(_proxy(engine, glm, _cfg()), allow_compaction=True)
    samples, _ = _round(tmp_path, recorder, engine, ResendingAgent(5), n=1)
    trial = _by_trial(samples)[0]
    assert len(trial) == 4
    assert [s["metadata"]["compaction"]["retry_replays"] for s in trial] == [1] * 4
    validate_compaction_group(samples)


def _segment(index: int, count: int, **meta) -> dict:
    compaction = {"schema": 1, "trajectory_id": "t", "segment_index": index, "num_segments": count,
                  "context_budget": 350, "summary_ok": True if index < count - 1 else None,
                  "kept_steps": None if index == 0 else 1, "segment_tokens": 5,
                  "tokens_after": 5 * (count - index - 1), "summary_tokens": 2 if index < count - 1 else 0,
                  "retry_replays": 0, **meta}
    return {"group_index": 0, "index": 0, "reward": 1.0, "status": "completed",
            "metadata": {"compaction": compaction}}


@pytest.mark.parametrize(("patch", "error"), [
    ({1: {"summary_tokens": -1}}, "summary_tokens"),
    ({1: {"summary_tokens": 1.5}}, "summary_tokens"),
    ({1: {"retry_replays": 2}}, "retry_replays"),
    ({1: {"kept_steps": None}}, "kept_steps"),
    ({0: {"summary_ok": None}}, "summary_ok"),
    ({0: {"summary_ok": 1}}, "summary_ok")])
def test_the_group_check_reads_the_new_fields_and_the_null_positions(patch, error):
    rows = [_segment(i, 3) for i in range(3)]
    validate_compaction_group(rows)  # null kept_steps on the first, null summary_ok on the last
    for i, fields in patch.items():
        rows[i]["metadata"]["compaction"].update(fields)
    with pytest.raises(ValueError, match=error):
        validate_compaction_group(rows)


def test_samples_without_the_new_fields_still_pass():
    rows = [_segment(i, 2) for i in range(2)]
    for row in rows:
        del row["metadata"]["compaction"]["summary_tokens"], row["metadata"]["compaction"]["retry_replays"]
    assert validate_compaction_group(rows)[0].num_segments == 2
