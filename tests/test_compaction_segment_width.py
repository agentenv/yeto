"""A compaction segment can be wider than ``context_budget``.

The records proxy checks the trigger before a model call. A step that does not
trigger still adds its reply and then a tool output of any length, and the summary
turn follows, so a non-last segment's stream can grow to the engine context
(``max_model_len``). With ``agentic.max_response_len == context_budget`` such a
segment is cut, loses its summary, and ``finish_sample`` drops the whole trial
(``compaction_segment_truncated``). The launch checks therefore ask for rows of
``max_model_len``: the spec of this case is refused, and with
``max_response_len = max_model_len`` the same trial trains.

Driven like tests/test_rl_compaction_summary_turn.py (GLM template from the local
HF cache, skipped if absent; a scripted FakeEngine).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from test_records_proxy import SCRIPT, _template
from test_rl_compaction_summary_turn import SUMMARY, prompts, proxy_for, run

from yeto.rl.agentic import AgentTrajectory, Outcome
from yeto.rl.agentic.admission import HarnessMetrics
from yeto.rl.agentic.finish import SEGMENT_TRUNCATED, finish_sample
from yeto.rl.compaction import config_from_spec
from yeto.rl.harness.gateway.recorder import segments_from_records

MAX_MODEL_LEN = 65536
T_COMP = 1000
RESULTS = ["result 0", "result 1 " + "word " * 3000, "result 2"]  # a big tool output after the 2nd call


@pytest.fixture(scope="module")
def glm():
    return _template("glm5")


def _spec(budget: int, max_response_len: int) -> dict[str, Any]:
    return {"engine": {"server": {"max_model_len": MAX_MODEL_LEN}},
            "agentic": {"turn_max_tokens": 256, "max_response_len": max_response_len,
                        "task_defaults": {"agent": "pi"},
                        "compaction": {"context_budget": budget, "t_comp": T_COMP, "summary_max_tokens": 64,
                                       "summary_prompt": "Summarize in <summary></summary>."}}}


def _budget(glm) -> int:
    """A budget the 2nd request stays below and the 3rd (after the big output) is above."""
    return prompts(glm, list(SCRIPT), RESULTS, max_model_len=MAX_MODEL_LEN)[1] + T_COMP


def _trial(glm, cfg) -> dict[str, Any]:
    proxy, _engine = proxy_for(glm, [SCRIPT[0], SCRIPT[1], SUMMARY, SCRIPT[2]], cfg, max_model_len=MAX_MODEL_LEN)
    run(proxy, "t", RESULTS)
    return proxy.records("t")


def _finish(records: dict[str, Any], cfg, max_response_len: int):
    class Recorder:
        async def close_segments(self, session):
            return segments_from_records(records, context_budget=cfg.context_budget)

    fields = {f.name: None for f in dataclasses.fields(AgentTrajectory)}
    fields.update(outcome=Outcome.OK, reward=1.0, reason=None)
    return asyncio.run(finish_sample(AgentTrajectory(**fields), Recorder(), "t", HarnessMetrics(),
                                     max_response_len=max_response_len, segmented=True))


def test_a_row_of_context_budget_is_refused_when_the_engine_context_is_wider(glm):
    budget = _budget(glm)
    with pytest.raises(ValueError, match=f"agentic.max_response_len {budget} < the engine context length "
                                         f"max_model_len {MAX_MODEL_LEN}"):
        config_from_spec(_spec(budget, budget))


def test_the_refused_case_is_real_a_segment_outgrows_context_budget(glm):
    """What the refused spec would do: segment 0 is wider than context_budget, and a
    row of context_budget drops the trial."""
    budget = _budget(glm)
    cfg = config_from_spec(_spec(budget, MAX_MODEL_LEN))
    records = _trial(glm, cfg)
    first, _second = records["sessions"][0]["samples"]
    assert [t["kind"] for t in first["turns"]][-1] == "summary"
    response = len(first["token_ids"]) - len(first["turns"][0]["prompt_added"])
    assert response > budget
    lost = _finish(records, cfg, max_response_len=budget)
    assert lost.trajectory.outcome is Outcome.ABORTED
    assert str(lost.trajectory.reason).startswith(SEGMENT_TRUNCATED)


def test_with_rows_of_max_model_len_the_same_trial_trains(glm):
    budget = _budget(glm)
    cfg = config_from_spec(_spec(budget, MAX_MODEL_LEN))
    kept = _finish(_trial(glm, cfg), cfg, max_response_len=MAX_MODEL_LEN)
    assert kept.trajectory.outcome is Outcome.OK
    assert len(kept.parts) == 2  # both segments train, the summary included
