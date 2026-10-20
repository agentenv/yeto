"""S26 E9.3: disjoint sampler data slots, a normal trainer end on an exhausted
buffer, and the launcher following a rescheduled Modal container's log."""

from __future__ import annotations

import io
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine import sample_roles
from yeto.rl.engine.buffer_payload import SampleBufferExhausted, SampleBufferRolloutError, is_buffer_exhausted
from yeto.rl.engine.engine_sampler_entry import JsonlPrompts, SamplerSpecError
from yeto.rl.engine.sample_roles import DataSlot


# ------------------------------------------------------------------ DataSlot
def test_data_slot_parse_and_rows():
    assert DataSlot.parse(None) == DataSlot() and DataSlot().whole
    slot = DataSlot.parse("1/3")
    assert (slot.index, slot.count, slot.env()) == (1, 3, "1/3")
    assert slot.rows(list(range(10))) == [1, 4, 7]
    for bad in ("3/3", "-1/2", "1/0", "2"):
        with pytest.raises(ValueError):
            DataSlot.parse(bad)
    with pytest.raises(ValueError, match="empty"):
        DataSlot(2, 3).rows([0, 1])


def test_slots_of_all_samplers_cover_the_data_once():
    rows = list(range(17))
    drawn = [r for k in range(3) for r in DataSlot(k, 3).rows(rows)]
    assert sorted(drawn) == rows


def test_pair_island_args_give_each_sampler_its_slot_and_flag():
    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:sampler", "2:trainer"],
                           rl_sampler_max_backlog_rounds=None, rl_sample_wait_s=None, total_steps=8,
                           rl_sampler_spot=False)
    a0, a1, t = (sample_roles.island_args(pair, m, 3) for m in range(3))
    assert (a0.rl_sampler_data_slot, a1.rl_sampler_data_slot) == ("0/2", "1/2")
    assert getattr(t, "rl_sampler_data_slot", None) is None
    assert " --rl-sampler-data-slot 1/2" in sample_roles.island_flags(a1)
    one = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:trainer"], rl_sampler_max_backlog_rounds=None,
                          rl_sample_wait_s=None, total_steps=8, rl_sampler_spot=False)
    assert sample_roles.island_args(one, 0, 2).rl_sampler_data_slot is None  # one sampler: the whole data


def test_data_slot_flag_on_both_layers():
    import argparse

    from yeto.rl.engine.run_flags import CLI, ISLAND, flags_for

    for layer in (CLI, ISLAND):
        parser = argparse.ArgumentParser()
        for flag in flags_for(layer):
            if flag.options[0] == "--rl-sampler-data-slot":
                parser.add_argument(*flag.options, **dict(flag.kwargs))
        assert parser.parse_args(["--rl-sampler-data-slot", "0/2"]).rl_sampler_data_slot == "0/2"


# ------------------------------------------------------------ engine sampler
def _prompts(tmp_path, n=10):
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps({"q": f"q{i}", "a": str(i)}) + "\n" for i in range(n)))
    return path


def test_engine_sampler_prompts_of_two_slots_are_disjoint(tmp_path):
    path = _prompts(tmp_path)
    ids = []
    for slot in ("0/2", "1/2"):
        prompts = JsonlPrompts(path, input_key="q", label_key="a", groups_per_round=2, data_slot=slot)
        ids.append([item.prompt_id for r in range(2) for item in prompts(r)])
    assert ids == [["row0", "row2", "row4", "row6"], ["row1", "row3", "row5", "row7"]]
    whole = JsonlPrompts(path, input_key="q", label_key="a", groups_per_round=2)
    assert [i.prompt_id for i in whole(1)] == ["row2", "row3"]  # unchanged without a slot
    with pytest.raises(SamplerSpecError, match="data_slot"):
        JsonlPrompts(path, input_key="q", label_key="a", groups_per_round=2, data_slot="2/2")


def test_stage1_sampler_spec_carries_the_slot_only_with_several_samplers():
    from yeto.rl import stage1

    src = open(stage1.__file__).read()
    assert "slot=DataSlot(samplers.index(node)" in src and '"data_slot": slot.env()' in src


# --------------------------------------------------------------- Miles source
class _FakeDataset:
    """The fields and shuffle of Miles' ``miles.utils.data.Dataset``."""

    def __init__(self, rows, seed=3):
        self.origin_samples = list(rows)
        self.samples = list(rows)
        self.seed, self.epoch_id = seed, -1

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __len__(self):
        return len(self.samples)


def _source(rows, shuffle):
    source = SimpleNamespace(dataset=_FakeDataset(rows), epoch_id=0, sample_offset=0, sample_group_index=0,
                             sample_index=0)
    if shuffle:
        source.dataset.shuffle(0)
    return source


@pytest.mark.parametrize("shuffle", [False, True])
def test_miles_data_source_slots_are_disjoint(shuffle):
    from yeto.rl.adapters.miles.buffer_rollout import apply_data_slot

    rows = list(range(20))
    drawn = []
    for k in range(2):
        source = _source(rows, shuffle)
        assert apply_data_slot(source, DataSlot(k, 2), shuffle=shuffle) == 10
        assert sorted(source.dataset.samples) == rows[k::2]
        drawn.append(set(source.dataset.samples))
    assert not drawn[0] & drawn[1] and drawn[0] | drawn[1] == set(rows)


def test_miles_data_slot_refuses_after_the_first_draw():
    from yeto.rl.adapters.miles.buffer_rollout import apply_data_slot

    source = _source(list(range(4)), False)
    source.sample_offset = 2
    with pytest.raises(SampleBufferRolloutError, match="before the first draw"):
        apply_data_slot(source, DataSlot(0, 2), shuffle=False)
    whole = _source(list(range(4)), False)
    assert apply_data_slot(whole, DataSlot(), shuffle=False) == 4


def test_island_env_exports_the_slot_for_a_sampler_only():
    from yeto.rl.adapters.miles import buffer_rollout as br

    env = br.island_env("sampler", store_root="/s", sampler_id="island0", shards=["island0", "island1"],
                        data_slot="0/2")
    assert env[br.DATA_SLOT_ENV] == "0/2"
    with pytest.raises(ValueError, match="only a sampler"):
        br.island_env("trainer", store_root="/s", max_policy_age=2, data_slot="0/2")


# ------------------------------------------------------- exhausted buffer end
def test_is_buffer_exhausted_follows_wrappers():
    inner = SampleBufferExhausted("ended")
    assert is_buffer_exhausted(inner)
    try:
        try:
            raise inner
        except SampleBufferExhausted as e:
            raise RuntimeError("ray task failed") from e
    except RuntimeError as outer:
        assert is_buffer_exhausted(outer)
    ray_like = RuntimeError("RayTaskError")
    ray_like.cause = inner
    assert is_buffer_exhausted(ray_like)
    dual = type("SampleBufferExhausted", (RuntimeError,), {})("dual class of another process")
    assert is_buffer_exhausted(dual)
    assert not is_buffer_exhausted(SampleBufferRolloutError("other"))
    assert not is_buffer_exhausted(None)


def test_batch_reader_raises_exhausted_on_an_ended_empty_buffer():
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader

    buffer = SimpleNamespace(read_cursor=5, data_ended=True, select=lambda consumer, n: [])
    reader = BatchReader(buffer, max_policy_age=2, wait_s=0, poll_s=0)
    with pytest.raises(SampleBufferExhausted, match="no trainable group"):
        reader.take(3, 4, current_version=3)


def _fake_driver(tmp_path, sync, exhaust_at):
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)}, step_delta=1.0, capability_profile="miles-like")
    generate = engine.rollout.generate

    def gen(rollout_id, **kw):
        if rollout_id == exhaust_at:
            raise RuntimeError("ray task failed") from SampleBufferExhausted("ended")
        return generate(rollout_id, **kw)

    engine.rollout.generate = gen
    return IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement, algorithm=AlgorithmSpec(), sync=sync,
                        events=EventTape(tmp_path / "e.jsonl", 0), capabilities=fake_capabilities("miles-like"))


def test_driver_ends_finished_when_the_buffer_is_exhausted(tmp_path):
    from yeto.rl.engine.bridges import LocalOnlySync

    final = _fake_driver(tmp_path, LocalOnlySync(5), exhaust_at=3).run()
    assert final.policy_version == 3
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    assert "rl_buffer_exhausted" in names and "island_leave" not in names
    assert next(e for e in events if e["event"] == "rl_buffer_exhausted")["rollout_id"] == 3


def test_driver_still_fails_on_other_errors_and_on_a_strict_roster(tmp_path):
    from yeto.rl.engine.bridges import LocalOnlySync, StrictAvgSync

    assert StrictAvgSync.ends_on_data_end is False
    sync = LocalOnlySync(5)
    sync.ends_on_data_end = False
    with pytest.raises(RuntimeError, match="ray task failed"):
        _fake_driver(tmp_path, sync, exhaust_at=2).run()


# ------------------------------------------------- rescheduled container log
def test_container_log_follower_streams_new_containers_only():
    from yeto.modal_runner import ContainerLogFollower

    lists = iter([{"ta-a"}, {"ta-a"}, {"ta-a", "ta-b"}, {"ta-b"}])
    streamed = []

    def stream(cid):
        streamed.append(cid)
        yield "[modal-island 1] rank 0 container ta-b\n"
        yield "round 2 done\nround 3 done\n"

    class Collector:
        def __init__(self):
            self.lines = []

        def feed(self, line):
            self.lines.append(line)

    collector = Collector()
    out = io.StringIO()
    follower = ContainerLogFollower(lambda: next(lists), stream, expected=1,
                                    route=lambda learner: ("island1", collector) if learner == 1 else None,
                                    out=out)
    assert follower.poll_once() == [] and follower.baseline == {"ta-a"}
    assert follower.poll_once() == []
    assert follower.poll_once() == ["ta-b"]
    follower.threads[0].join(5)
    assert follower.poll_once() == []  # followed once
    text = out.getvalue()
    assert "new Modal container ta-b" in text
    assert "[island1] round 2 done" in text and "[island1] round 3 done" in text
    assert streamed == ["ta-b"] and collector.lines[-1] == "round 2 done\nround 3 done"


def test_container_log_follower_only_when_new_containers_are_expected():
    from yeto.launch.modal_islands import container_log_follower_for

    ops = SimpleNamespace(list_container_ids=lambda: set(), stream_container_logs=lambda cid: iter(()))
    cfgs = {"island0": SimpleNamespace(num_nodes=1, learner_id=0)}
    assert container_log_follower_for(None, ops, cfgs, None, poller=object(), start=False) is None
    assert container_log_follower_for(None, None, cfgs, None, poller=None, start=False) is None
    follower = container_log_follower_for(None, ops, cfgs, None, poller=None, start=False)
    assert follower is not None and follower.expected_count == 1
    assert container_log_follower_for(None, SimpleNamespace(list_container_ids=lambda: set()), cfgs, None,
                                      poller=None, start=False) is None
