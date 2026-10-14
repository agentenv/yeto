"""P0-S: prompts of Miles fully_async rollouts are accounted for and, at a
cut, returned to the data buffer so the resumed run generates them again."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import fully_async_prompts as prompts
from yeto.rl.adapters.miles.in_flight import export_in_flight


class Sample:
    def __init__(self, group_index: int) -> None:
        self.group_index = group_index
        self.index = group_index
        self.response = ""
        self.tokens = [1]
        self.response_length = 0
        self.metadata = {}

    def reset_for_retry(self) -> None:
        self.response = ""

    def to_dict(self) -> dict:
        return {"group_index": self.group_index, "index": self.index, "response": self.response}


class DataSource:
    """Miles' data source with a reuse buffer: buffer first, then the dataset."""

    def __init__(self) -> None:
        self.offset = 0
        self.buffer: list[list[Sample]] = []

    def get_samples(self, n: int) -> list[list[Sample]]:
        out = []
        for _ in range(n):
            if self.buffer:
                out.append(self.buffer.pop(0))
            else:
                out.append([Sample(self.offset)])
                self.offset += 1
        return out

    def add_samples(self, groups) -> None:
        self.buffer.extend(groups)


@pytest.fixture
def executor():
    executor = SimpleNamespace(data_source=DataSource(), args=SimpleNamespace(fully_async=True),
                               generate_rollout=SimpleNamespace(_worker=None, _output=None))
    prompts.install(executor)
    yield executor
    prompts._LEDGER = None


def test_round_counts_report_draws_and_miles_stale_drops(executor):
    ledger = prompts.current()
    executor.data_source.get_samples(3)
    executor.generate_rollout._output = SimpleNamespace(_metric_stale_groups=2)
    assert ledger.take_round_counts() == {"submitted_groups": 3, "stale_dropped_groups": 2}
    executor.generate_rollout._output = SimpleNamespace(_metric_stale_groups=5)
    assert ledger.take_round_counts() == {"submitted_groups": 0, "stale_dropped_groups": 3}


def test_an_unexposed_stale_count_is_reported_unknown(executor, caplog):
    counts = prompts.current().take_round_counts()
    assert counts["stale_dropped_groups"] is None
    assert "reported unknown" in caplog.text


def test_cut_returns_every_untrained_prompt_and_stops_the_worker(executor):
    source, ledger = executor.data_source, prompts.current()
    drawn = source.get_samples(5)  # in flight / waiting in Miles' output buffer
    ledger.mark_trained(drawn[:2])  # trained this round
    source.add_samples([drawn[2]])  # stale: Miles recycled it into the buffer
    drawn[3][0].response = "partial"  # generation in progress

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    worker = asyncio.run_coroutine_threadsafe(asyncio.sleep(3600), loop)
    executor.generate_rollout._worker = asyncio.run_coroutine_threadsafe(
        _task_of(worker), loop).result()
    try:
        exported = export_in_flight(executor)
    finally:
        loop.call_soon_threadsafe(loop.stop)

    assert executor.generate_rollout._worker is None  # the next call starts a fresh worker
    assert source.offset == 5  # the cursor is not rewound: the prompts are in the buffer
    assert sorted(g[0].group_index for g in source.buffer) == [2, 3, 4]
    assert all(g[0].response == "" for g in source.buffer)  # reset for a fresh generation
    assert drawn[3][0].response == "partial"  # a running task keeps its own sample
    assert exported["buffer_groups"] == 3
    assert sorted({e["group_id"] for e in exported["entries"]}) == ["g2", "g3", "g4"]
    # the resumed run draws exactly the untrained prompts again
    assert sorted(g[0].group_index for g in source.get_samples(3)) == [2, 3, 4]


async def _task_of(future):
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            return task
    raise AssertionError("no worker task")


def test_cut_without_the_ledger_is_refused():
    executor = SimpleNamespace(data_source=DataSource(), args=SimpleNamespace(fully_async=True),
                               generate_rollout=None)
    with pytest.raises(prompts.PromptLedgerError, match="not tracked"):
        export_in_flight(executor)


def test_overlap_round_report_carries_the_ledger_counts(monkeypatch):
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    monkeypatch.setattr(hook, "build_metadata", lambda args, samples: {
        "completed": 2, "filtered": 0, "max_policy_age": 1})
    sent = []
    monkeypatch.setattr(hook, "put_to_sink", sent.append)
    hook.extract_rollout_metadata(
        SimpleNamespace(), [], round_counts={"submitted_groups": 3, "stale_dropped_groups": None})
    assert sent[0]["submitted_groups"] == 3 and sent[0]["stale_dropped_groups"] is None
    assert "aborted_in_flight_groups" not in sent[0] and "carried_out_groups" not in sent[0]
