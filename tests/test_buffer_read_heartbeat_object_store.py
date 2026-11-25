"""A buffer trainer's read heartbeat (``sample-buffer/read-position.json``) on an object store.

A verl buffer trainer on a staged S3 store wrote the heartbeat as a local file under the
stage root, into a ``sample-buffer/`` directory nobody had created: every consumed group
logged ``rl_buffer_read_position_failed`` (an L40S run), and even written, a local file on the
trainer's disk is no heartbeat for a sampler on another host. It now goes where the samplers
read it, as on the Miles trainer: the object on an object store, else the file, whose
directory is created first.
"""

from __future__ import annotations

from pathlib import Path

from test_rl_object_store import FakeS3
from test_rl_verl_buffer_trainer import SHAPE, FakeBufferActor, _sample
from yeto.cloud import s3_store
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore, store_for
from yeto.rl.engine.sample_buffer import (BUFFER_DIR, READ_POSITION, READ_POSITION_KEY, FileSampleBuffer,
                                          open_trainer_buffer, read_buffer_read_position, read_read_position,
                                          write_read_position)

MOUNT = "~/yeto-checkpoint-store/runs/a"


def _staged(tmp_path, s3, stage):
    """The run store as one host sees it: staged on its own disk, reached by the object API."""
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: "s3://bkt/runs/a",
           s3_store.STAGE_ENV: str(tmp_path / stage)}
    return store_for(MOUNT, environ=env, client=s3)


def _verl_trainer(tmp_path, store, events):
    from yeto.rl.adapters.verl.buffer_ports import install_buffer_ports
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland

    island = FullyAsyncIsland(FakeBufferActor(), learner_id=0, limit=1, required=1,
                              out_dir=str(tmp_path / "out"), readback_dir=str(tmp_path / "rb"))
    (tmp_path / "out").mkdir(exist_ok=True)
    island.emit = lambda event, **f: events.append((event, f))
    buffer = open_trainer_buffer(store, batch_size=1, create_index=False, shards=["pool-0"])
    install_buffer_ports(island, buffer=buffer, releases=PolicyReleaseStore(store), shape=SHAPE,
                         base_dir=tmp_path, poll_s=0.01, wait_s=5)
    island.consume_map.record(0, 0)  # v0 published: the groups' behavior version is known
    island.published = (0, "0" * 64)
    return island


def test_a_verl_buffer_trainer_on_a_staged_s3_store_writes_its_heartbeat_where_samplers_read_it(tmp_path):
    s3 = FakeS3()
    sampler_store = _staged(tmp_path, s3, "sampler-host")
    writer = GroupWriter(FileSampleBuffer(sampler_store, batch_size=1, index="shards/pool-0.jsonl"), "pool-0")
    for key in ("g0", "g1"):
        writer.write([_sample(0, index=0), _sample(0, index=1)], group_key=key)
    events = []
    island = _verl_trainer(tmp_path, _staged(tmp_path, s3, "trainer-host"), events)
    for _ in range(2):  # two groups kept, recorded at the round's assemble (one heartbeat each)
        assert island.call("yeto_pull_sample") is not None
        island.call("yeto_keep_last")
    island.call("yeto_assemble")
    assert not [f for e, f in events if e == "rl_buffer_read_position_failed"], events
    # the heartbeat is the object samplers read, not a file on the trainer's disk
    assert ("bkt", f"runs/a/{READ_POSITION_KEY}") in s3.objects
    assert not (tmp_path / "trainer-host" / "bkt" / "runs" / "a" / BUFFER_DIR / READ_POSITION).exists()
    beat = read_buffer_read_position(FileSampleBuffer(_staged(tmp_path, s3, "other-host"), batch_size=1,
                                                      create_index=False))
    assert beat is not None and beat["buffer_selected"] == 2 and beat["buffer_shards"] == {"pool-0": 2}


def test_the_miles_trainer_heartbeat_on_a_staged_s3_store_is_the_object_too(tmp_path):
    """The Miles trainer already wrote it through the store (no local directory needed)."""
    from yeto.rl.adapters.miles.buffer_rollout import trainer_read_heartbeat

    s3 = FakeS3()
    buffer = open_trainer_buffer(_staged(tmp_path, s3, "trainer-host"), batch_size=1, create_index=False,
                                 shards=["pool-0"])
    assert trainer_read_heartbeat(buffer)(3, {"pool-0": 3}) == READ_POSITION_KEY
    assert read_buffer_read_position(FileSampleBuffer(_staged(tmp_path, s3, "other-host"), batch_size=1,
                                                      create_index=False))["buffer_selected"] == 3


def test_a_heartbeat_file_creates_its_directory(tmp_path):
    """A trainer that opens its buffer without creating the index (create_index=False) has no
    sample-buffer/ directory yet on a plain directory store."""
    root = tmp_path / "store" / BUFFER_DIR
    assert not root.exists()
    path = write_read_position(root, buffer_selected=4, buffer_shards=None, train_step=1, release=None)
    assert Path(path) == root / READ_POSITION and read_read_position(root)["buffer_selected"] == 4


def test_a_verl_buffer_trainer_on_a_plain_directory_writes_the_heartbeat_file(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    GroupWriter(FileSampleBuffer(store, batch_size=1, index="shards/pool-0.jsonl"), "pool-0").write(
        [_sample(0, index=0), _sample(0, index=1)], group_key="g0")
    events = []
    island = _verl_trainer(tmp_path, store, events)
    island.call("yeto_pull_sample")
    island.call("yeto_keep_last")
    island.call("yeto_assemble")
    assert not [f for e, f in events if e == "rl_buffer_read_position_failed"]
    assert read_read_position(tmp_path / "store" / BUFFER_DIR)["buffer_selected"] == 1
