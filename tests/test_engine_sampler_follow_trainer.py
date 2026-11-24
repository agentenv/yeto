"""An engine sampler that follows one trainer (an external pool bound to trainer k of
a fleet of islands): it runs until that trainer ended (``until_trainers_end``, the
end records in its own store) and its start event names its ``island`` and
``store``. CPU, the fake engine; no cloud resource."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_stage1_gaps import _sampler_spec
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_resume import TRAINERS_ENDED_DIR, TRAINERS_ENDED_EVENT, write_trainer_end

pytestmark = pytest.mark.usefixtures("ports_v1_pin")

# end_trainer_reward writes a trainer end record into these stores (os.pathsep separated)
# on its first call; the sampler imports the reward by name, so it reads the environment
END_STORES_ENV = "YETO_TEST_END_TRAINER_STORES"


def end_trainer_reward(item, completion) -> float:
    """Scores 0.5; on its first call (the first round's first completion) the
    trainers of :data:`END_STORES_ENV` write their end records."""
    stores = os.environ.pop(END_STORES_ENV, "")
    for store in filter(None, stores.split(os.pathsep)):
        write_trainer_end(CheckpointStore(Path(store)), "0")
    return 0.5


def _end_on_first_score(monkeypatch, *stores: Path) -> None:
    monkeypatch.setenv(END_STORES_ENV, os.pathsep.join(map(str, stores)))


def _events(tmp_path):
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    return [json.loads(line) for line in tape.read_text().splitlines()]


def _groups(store: Path) -> int:
    reader = FileSampleBuffer(CheckpointStore(store), batch_size=1)
    reader.refresh()
    return len(reader.select("t", 100))


def _follower(tmp_path, island=1, **over):
    root = tmp_path / "store"
    follow = {"store": str(root / f"trainer{island}"), "island": island, "until_trainers_end": 1,
              "num_rollouts": None, "reward": "tests.test_engine_sampler_follow_trainer:end_trainer_reward",
              "poll_s": 0.01}
    return _sampler_spec(tmp_path, **{**follow, **over}), root


# ------------------------------------------------ 1.4: until_trainers_end
def test_the_sampler_writes_the_round_in_flight_then_leaves_once_its_trainer_ended(tmp_path, monkeypatch):
    spec, root = _follower(tmp_path)
    _end_on_first_score(monkeypatch, root / "trainer1")  # mid round 0: its trainer ends
    assert entry.run(spec, resource_sampler=lambda emit: None) == 1
    assert _groups(root / "trainer1") == 2  # round 0's two groups, all written
    events = _events(tmp_path)
    names = [e["event"] for e in events]
    assert names.count("rl_sampler_engine_round") == 1
    (ended,) = [e for e in events if e["event"] == TRAINERS_ENDED_EVENT]
    assert ended["trainers"] == 1 and ended["rounds"] == 1
    (leave,) = [e for e in events if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FINISHED"


def test_only_the_records_of_its_own_store_stop_it(tmp_path, monkeypatch):
    # trainer 0 ends mid round 0; the pool of trainer 1 goes on to its round limit
    spec, root = _follower(tmp_path, num_rollouts=3)
    _end_on_first_score(monkeypatch, root / "trainer0", root)
    assert entry.run(spec, resource_sampler=lambda emit: None) == 3
    assert (root / "trainer0" / TRAINERS_ENDED_DIR / "0.json").exists()
    assert (root / TRAINERS_ENDED_DIR / "0.json").exists()
    assert not (root / "trainer1" / TRAINERS_ENDED_DIR).exists()
    assert not [e for e in _events(tmp_path) if e["event"] == TRAINERS_ENDED_EVENT]


def test_a_trainer_that_ended_before_the_start_runs_no_round(tmp_path):
    spec, root = _follower(tmp_path)
    write_trainer_end(CheckpointStore(root / "trainer1"), "1")
    assert entry.run(spec, resource_sampler=lambda emit: None) == 0
    assert [e["event"] for e in _events(tmp_path)].count(TRAINERS_ENDED_EVENT) == 1


def test_a_failed_listing_counts_as_not_ended_and_is_reported_once():
    calls, events = [], []

    class Store:
        def list(self, prefix):
            calls.append(prefix)
            if len(calls) <= 2:
                raise OSError("store down")
            return ["0.json"]

    watch = entry.TrainersEndWatch(Store(), 1, lambda event, **f: events.append((event, f)))
    assert watch() is False and watch() is False and watch() is True and watch() is True
    assert len(calls) == 3  # sticky once ended
    assert [e for e, _ in events] == ["rl_sampler_store_read_failed", "rl_sampler_store_read_ok"]
    assert events[1][1]["failed_reads"] == 2


@pytest.mark.parametrize(("over", "match"), [
    ({"until_trainers_end": -1}, "until_trainers_end must be an integer >= 0"),
    ({"until_trainers_end": True}, "until_trainers_end must be an integer >= 0"),
    ({"island": "1"}, "island must be a learner id"),
    ({"island": -1}, "island must be a learner id"),
    ({"until_trainers_end": 1, "data_path": "syncer", "syncer": "h:1", "channel": 0, "work_dir": "/w"},
     "needs the r2 data path"),
    ({"num_rollouts": None}, "num_rollouts is required"),
])
def test_spec_values_are_checked(tmp_path, over, match):
    with pytest.raises(entry.SamplerSpecError, match=match):
        entry.island_config(_sampler_spec(tmp_path, **over))


def test_without_a_round_limit_a_follower_has_no_round_limit(tmp_path):
    from yeto.rl.engine.rollout_data_path import NO_LIMIT_ROUNDS

    spec, _root = _follower(tmp_path)
    assert entry.island_config(spec).num_rollouts == NO_LIMIT_ROUNDS
    assert entry.island_config({**spec, "num_rollouts": 5}).num_rollouts == 5


# ------------------------------------------------ 1.5: island and store on rl_sampler_start
def test_the_start_event_names_the_island_and_the_store(tmp_path):
    spec, root = _follower(tmp_path, num_rollouts=1)
    entry.run(spec, resource_sampler=lambda emit: None)
    (start,) = [e for e in _events(tmp_path) if e["event"] == "rl_sampler_start"]
    assert start["island"] == 1 and start["store"] == str(root / "trainer1")


def test_a_single_trainer_sampler_start_event_only_gains_island_and_store(tmp_path):
    from yeto.rl.engine.engine_sampler import START_EVENT

    entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None)
    (start,) = [e for e in _events(tmp_path) if e["event"] == START_EVENT]
    assert start["island"] is None and start["store"] == str(tmp_path / "store")
    # the fields of the start event before this change, plus the tape head
    before = {"sampler_id", "num_rollouts", "max_backlog_groups", "sampling", "sampling_digest",
              "allow_penalties", "max_prompt_tokens", "provenance", "loop_guard"}
    assert set(start) - {"event", "time_unix"} == before | {"island", "store"}
