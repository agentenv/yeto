"""A sampler counts its backlog from the furthest synced-trainer lane.

Modal (P8, 2 synced trainers on lanes 0/1 of one buffer, 2 samplers): trainer 1
was stopped; its lane's groups stayed unread and, counted from the release's read
position, filled the samplers' backlog cap. The samplers stopped, trainer 0 read
its lane dry and waited, and the merged version that would release a new policy
never came. Counted from the furthest lane the samplers keep feeding trainer 0.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.merged_release import (
    BufferLane, LaneCursorLog, ObjectLaneCursorLog, fastest_lane_read, lane_cursor_log)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import BUFFER_DIR, FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

GROUPS = 2  # groups per trainer round


def _write(buffer, n, start=0):
    for i in range(start, start + n):
        ref = buffer.write_payload(f"g{i}", br.encode_group([{"tokens": [i]}]))
        buffer.append(SampleIndexEntry(f"g{i}", 0, ref))


def _sampler(buffer, cap, selected=0):
    seen = SimpleNamespace(buffer_shards=None, buffer_selected=selected, seq=1, policy_hash="h",
                           released_at=0.0)
    return SamplerLoop(follower=SimpleNamespace(last_seen=seen), buffer=buffer, generate=lambda r: None,
                       num_rollouts=1, max_backlog_groups=cap, emit=lambda *a, **k: None)


def test_a_stopped_trainers_lane_does_not_starve_the_live_one(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    writer = FileSampleBuffer(store, batch_size=1)
    readers = []
    for m in range(2):
        lane_buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=GROUPS,
                                       lane=BufferLane(m, 2))
        readers.append(br.BatchReader(lane_buffer, max_policy_age=2, wait_s=1.0, poll_s=0.01,
                                      cursor_log=br.open_lane_log(lane_buffer)))
    cap = 8
    sampler = _sampler(FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1), cap)
    _write(writer, cap)  # the sampler is at its cap before anyone reads
    assert sampler.backlog() == cap
    # both trainers take a round, then trainer 1 is gone; trainer 0 goes on as long
    # as the sampler keeps writing whenever its backlog is below the cap
    for m in (0, 1):
        readers[m].take(0, GROUPS, current_version=None)
    written, rounds = cap, 0
    for rollout_id in range(1, 12):
        while sampler.backlog() < cap:
            _write(writer, 1, start=written)
            written += 1
        groups, _report = readers[0].take(rollout_id, GROUPS, current_version=None)
        assert len(groups) == GROUPS
        rounds += 1
    assert rounds == 11
    # without the lane positions the backlog (from the release) would be at the cap
    assert len(writer) - 0 >= cap and sampler.backlog() <= cap


def test_fastest_lane_read_on_a_directory_and_an_object_store(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    assert fastest_lane_read(SimpleNamespace(store=store)) is None
    LaneCursorLog(tmp_path / "s" / BUFFER_DIR, BufferLane(0, 2)).append(0, [0], 4, shards={"a": 2, "b": 2})
    LaneCursorLog(tmp_path / "s" / BUFFER_DIR, BufferLane(1, 2)).append(0, [1], 9, shards={"a": 5, "b": 4})
    (tmp_path / "s" / BUFFER_DIR / "lanes" / "1-of-2.jsonl").open("a").write('{"torn')
    buffer = SimpleNamespace(store=store)
    assert fastest_lane_read(buffer) == 9
    assert fastest_lane_read(buffer, "a") == 5 and fastest_lane_read(buffer, "zz") is None

    class Objects:
        def __init__(self):
            self.data = {}

        def list(self, prefix):
            return [k for k in self.data if k.startswith(prefix)]

        def read(self, key):
            return self.data[key]

        def write(self, key, raw):
            self.data[key] = raw

    objects = Objects()
    for m, cursors in ((0, (2, 6)), (1, (3,))):
        log = ObjectLaneCursorLog(objects, BUFFER_DIR, BufferLane(m, 2))
        for k, cursor in enumerate(cursors):
            log.append(k, [k], cursor, shards={"a": cursor})
    assert fastest_lane_read(SimpleNamespace(store=objects)) == 6
    assert fastest_lane_read(SimpleNamespace(store=objects), "a") == 6
    assert json.loads(objects.data[sorted(objects.data)[0]])["shards"] == {"a": 2}


def test_a_lane_without_shard_positions_is_ignored_for_a_shard(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    log = lane_cursor_log(SimpleNamespace(store=store, lane=BufferLane(0, 1)))
    log.append(0, [0], 7)  # an old-format record: no shards
    assert fastest_lane_read(SimpleNamespace(store=store), "a") is None
    assert fastest_lane_read(SimpleNamespace(store=store)) == 7
