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
    output = SimpleNamespace(_metric_stale_groups=9)
    executor.generate_rollout._output = output
    assert ledger.take_round_counts() == {"submitted_groups": 3, "stale_dropped_groups": 9}


def test_stale_drops_follow_miles_per_step_reset(executor):
    """Miles' get_metrics zeroes the counter after each round's hook (G2 ps1:
    a running-total delta turned 9, 0, 10, 0 into 9, -9, 10, -10)."""
    ledger = prompts.current()
    output = SimpleNamespace(_metric_stale_groups=0)
    executor.generate_rollout._output = output
    reported = []
    for dropped in (9, 0, 10, 0):
        output._metric_stale_groups = dropped
        reported.append(ledger.take_round_counts()["stale_dropped_groups"])
        output._metric_stale_groups = 0  # Miles' get_metrics, right after the hook
    assert reported == [9, 0, 10, 0]


def test_an_unexposed_stale_count_is_reported_unknown(executor, caplog):
    counts = prompts.current().take_round_counts()
    assert counts["stale_dropped_groups"] is None
    assert "reported unknown" in caplog.text


def test_cut_records_every_untrained_prompt_and_keeps_the_worker_running(executor):
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
        # E1: the worker keeps generating ahead across the cut
        assert executor.generate_rollout._worker is not None
        assert not executor.generate_rollout._worker.done()
    finally:
        loop.call_soon_threadsafe(loop.stop)

    # the live run is untouched: same buffer, same in-flight samples
    assert source.offset == 5 and [g[0].group_index for g in source.buffer] == [2]
    assert drawn[3][0].response == "partial"
    assert sorted(prompts.current().untrained) == [3, 4]
    # the cut carries every drawn, untrained prompt, reset for a fresh generation
    assert exported["buffer_groups"] == 3
    assert sorted({e["group_id"] for e in exported["entries"]}) == ["g2", "g3", "g4"]
    assert all(e["engine_state"]["miles_sample"]["response"] == "" for e in exported["entries"])


def test_cut_reads_cursor_and_prompts_on_the_worker_loop(executor):
    """The cursor and the untrained prompts come from one read on the loop the worker
    draws on: a draw can never fall between them."""
    source = executor.data_source
    for name in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
        setattr(source, name, 0)
    real_get = prompts.current()._get_samples  # the unwrapped DataSource.get_samples

    def get_samples(n):
        groups = real_get(n)
        source.sample_offset += n
        source.sample_group_index += n
        source.sample_index += n
        return groups

    prompts.current()._get_samples = get_samples
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    seen = []

    async def drawing_worker():
        while True:
            source.get_samples(1)  # the ledger's wrapper (records the draw)
            await asyncio.sleep(0)

    executor.generate_rollout._worker = asyncio.run_coroutine_threadsafe(
        _start(drawing_worker), loop).result()
    try:
        for _ in range(20):
            exported = export_in_flight(executor)
            seen.append((exported["data_cursor"]["sample_offset"], exported["buffer_groups"]))
    finally:
        executor.generate_rollout._worker.get_loop().call_soon_threadsafe(
            executor.generate_rollout._worker.cancel)
        loop.call_soon_threadsafe(loop.stop)
    # nothing is trained: every drawn prompt is in the cut, and the cursor is just past it
    assert all(offset == groups for offset, groups in seen)
    assert seen[-1][0] > seen[0][0]  # the worker kept drawing between cuts


async def _start(fn):
    return asyncio.create_task(fn())


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
    # the round overlapped training: the carry-over event must not say serial
    assert sent[0]["execution_mode"] == "partitioned-overlap"
    from yeto.rl.adapters.miles.rollout import _carry_over_fields

    assert _carry_over_fields(sent[0])["execution_mode"] == "partitioned-overlap"


class _FullyAsyncFn:
    """Miles FullyAsyncRolloutFn's shape: a worker that submits ``_generate_group``
    tasks and waits on them with asyncio.wait (which never cancels them)."""

    def __init__(self) -> None:
        self._worker = None
        self._output = None
        self.ended: list[int] = []

    async def _generate_group(self, index: int) -> None:
        try:
            await asyncio.sleep(3600)  # a long agentic trajectory
        finally:
            self.ended.append(index)  # the agent's finally ends its trajectory

    async def worker_loop(self, groups: int) -> None:
        active = {asyncio.create_task(self._generate_group(i)) for i in range(groups)}
        while True:
            await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)


def test_cut_cancels_the_generation_the_stopped_worker_submitted(executor):
    """S24 T-TB2 j: stopping only the worker left its group tasks generating while the
    next worker submitted a full budget again (in flight 29 -> 146 over four cuts)."""
    rollout_fn = _FullyAsyncFn()
    other = _FullyAsyncFn()  # another rollout function's generation is not touched
    executor.generate_rollout = rollout_fn
    executor.data_source.get_samples(3)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def start():
        rollout_fn._worker = asyncio.create_task(rollout_fn.worker_loop(3))
        foreign = asyncio.create_task(other._generate_group(99))
        await asyncio.sleep(0)
        return foreign

    foreign = asyncio.run_coroutine_threadsafe(start(), loop).result()
    try:
        assert prompts.requeue_untrained(executor, timeout_s=5.0) == 3
        assert sorted(rollout_fn.ended) == [0, 1, 2]
        assert not foreign.done() and other.ended == []
    finally:
        async def stop_foreign():
            foreign.cancel()
            await asyncio.wait({foreign})

        asyncio.run_coroutine_threadsafe(stop_foreign(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
    assert rollout_fn._worker is None
