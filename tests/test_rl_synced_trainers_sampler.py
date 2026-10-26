"""S25 C3 end to end on CPU: two synced trainer islands, the real Rust syncer
(elastic OuterStep merge), one sampler that follows the releases.

The trainers are fake ports islands (``yeto.rl.engine.fake``) wired with the
production publisher wrappers (``sampler_island.wire_policy_releases``) and the
production lane reader (``buffer_rollout.BatchReader`` over a lane of one
``FileSampleBuffer``). The sampler is a thread: ``PolicyFollower`` loads each
release and writes groups stamped with the version it serves.

Checks (the same as the GPU run's, SESSION25/gpu-review/T-C3.md):
1. both trainers apply the same merged policy hash at every outer version;
2. the sampler loads exactly the merged versions 0..R, in order, each with the
   merged hash, all released by learner 0;
3. the two lanes are disjoint and no buffer position is trained twice;
4. the syncer ran R OuterSteps.

Runs when YETO_TEST_SYNCER points to a syncer binary (as
test_rl_island_pool_e2e.py)."""

from __future__ import annotations

from dataclasses import replace

import asyncio
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_engine_driver import NAME, _driver, _engine, _strict_config
from test_rl_island_pool_e2e import _free_port
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.sampler_island import wire_policy_releases
from yeto.rl.engine.bridges import PoolAvgSync
from yeto.rl.engine.merged_release import BufferLane
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sample_roles import SampleBufferConfig
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

ROUNDS = 3
MAX_POLICY_AGE = 2
_EXECUTOR_LOCK = threading.Lock()


class _LaneRollout:
    """The fake rollout pool of one trainer: each round first takes its groups
    from the island's buffer lane (as ``BufferReaderRolloutFn``), then lets the
    fake pool build the batch. ``_in_executor`` runs a function as in the Miles
    rollout executor, with this island's reader active."""

    def __init__(self, pool, reader: br.BatchReader, groups: int) -> None:
        self.pool, self.reader, self.groups = pool, reader, groups
        self.taken: list[tuple[int, list[int], dict[int, int]]] = []

    def generate(self, rollout_id, **kwargs):
        groups, report = self.reader.take(rollout_id, self.groups, current_version=None)
        self.taken.append((rollout_id, [p for p, _ in groups], report.behavior_versions))
        return self.pool.generate(rollout_id, **kwargs)

    def _in_executor(self, fn, *args):
        with _EXECUTOR_LOCK:
            br._ACTIVE[br.TRAINER] = self.reader
            return fn(None, *args)

    def __getattr__(self, name):
        return getattr(self.pool, name)


class _Actor:
    """``actor_model.export_hf``: the full weights the engine serves."""

    def __init__(self, engine) -> None:
        self.engine = engine

    async def export_hf(self, *, rollout_id: int, path: str) -> None:
        torch.save(self.engine.published_tensors, Path(path) / "model.pt")


def _sampler(store_root: Path, stop: threading.Event, loads: list, groups_ahead: int) -> None:
    store = CheckpointStore(store_root)
    buffer = FileSampleBuffer(store, batch_size=1)
    follower = PolicyFollower(
        PolicyReleaseStore(store),
        lambda release, directory: loads.append(
            (release.version, release.policy_hash, release.origin,
             torch.load(directory / "model.pt")[NAME].tolist())))
    n = 0
    while not stop.is_set():
        follower.refresh()
        release = follower.release
        if release is None or len(buffer) - int(release.buffer_selected or 0) >= groups_ahead:
            time.sleep(0.005)
            continue
        ref = buffer.write_payload(f"g{n}", br.encode_group([{"tokens": [n]}]))
        buffer.append(SampleIndexEntry(f"g{n}", follower.version, ref))
        n += 1
    buffer.mark_data_end()


def _events(path: Path, kind: str) -> list[dict]:
    return [r for r in (json.loads(line) for line in path.read_text().splitlines())
            if r.get("event") == kind]


@pytest.mark.skipif(not os.environ.get("YETO_TEST_SYNCER"),
                    reason="needs YETO_TEST_SYNCER (syncer binary)")
def test_two_synced_trainers_feed_one_sampler_through_merged_releases(tmp_path, monkeypatch):
    port, key = _free_port(), "c3-e2e"
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", key)  # the islands sign their HELLO
    out = Path(os.environ.get("YETO_TEST_ELASTIC_OUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()
    store_root = out / "store"
    store_root.mkdir(exist_ok=True)
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", str(ROUNDS), "--event-tape", str(out / "syncer-tape.jsonl"),
         "--quorum", "1", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-base-lag", "2", "--pipeline", "1",
         "--learner-weight", "tokens2-over-steps", "--island-lease-s", "6", "--syncer-epoch", "5"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT)
    stop, loads = threading.Event(), []
    try:
        time.sleep(1.0)
        engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
        sampler = threading.Thread(target=_sampler, daemon=True,
                                   args=(store_root, stop, loads, 4 * engines[0].groups))
        sampler.start()
        drivers, rollouts = [], []
        for i, engine in enumerate(engines):
            cfg = replace(_strict_config(out, engine, learner_id=i, rounds=ROUNDS, tape=f"island-{i}.jsonl"),
                          syncer_addr=("127.0.0.1", port), wan_streams=1)
            sync = PoolAvgSync(cfg, base_wait_s=30.0, syncer_epoch=5, lease_s=6.0)
            driver = _driver(engine, sync, out,
                             name=f"events-{i}.jsonl", learner_id=i)
            lane = BufferLane(i, 2)
            buffer = FileSampleBuffer(CheckpointStore(store_root), batch_size=engine.groups, lane=lane)
            reader = br.BatchReader(buffer, max_policy_age=MAX_POLICY_AGE, wait_s=30.0, poll_s=0.01,
                                    cursor_log=br.open_lane_log(buffer))
            rollout = _LaneRollout(driver.rollout, reader, engine.groups)
            driver.rollout = rollout
            wire_policy_releases(
                driver, actor_model=_Actor(engine), runner=SimpleNamespace(run=asyncio.run),
                miles_args=SimpleNamespace(
                    yeto_rl_sample_buffer=SampleBufferConfig(role="trainer", store=str(store_root)),
                    yeto_rl_sample_lane=lane, lora_rank=0))
            drivers.append(driver)
            rollouts.append(rollout)
        results, errors = {}, {}

        def run(i, d):
            try:
                results[i] = d.run()
            except BaseException as error:  # noqa: BLE001
                errors[i] = error

        threads = [threading.Thread(target=run, args=(i, d), daemon=True)
                   for i, d in enumerate(drivers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()
        assert errors == {}
        assert syncer.wait(15) == 0
    finally:
        stop.set()
        if syncer.poll() is None:
            syncer.kill()

    # 1. the same merged policy on both trainers at every outer version
    applied = [{e["sync/outer_version"]: e["sync/global_policy_hash"]
                for e in _events(out / f"events-{i}.jsonl", "rl_policy_apply")} for i in range(2)]
    assert sorted(applied[0]) == list(range(ROUNDS + 1)) and applied[0] == applied[1]
    assert torch.equal(results[0].tensors[NAME], results[1].tensors[NAME])
    # 2. the sampler loaded the merged versions in order, from the publisher only
    assert [v for v, *_ in loads] == list(range(ROUNDS + 1))
    assert all(h == applied[0][v] for v, h, *_ in loads)
    assert {o["learner_id"] for _, _, o, _ in loads} == {0}
    assert loads[-1][3] == results[0].tensors[NAME].tolist()
    released = _events(out / "events-0.jsonl", "rl_policy_released")
    assert [e["policy_version"] for e in released] == list(range(ROUNDS + 1))
    assert _events(out / "events-1.jsonl", "rl_policy_released") == []
    # 3. disjoint lanes, no position trained twice
    positions = [[p for _, ps, _ in r.taken for p in ps] for r in rollouts]
    assert all(p % 2 == i for i, ps in enumerate(positions) for p in ps)
    flat = positions[0] + positions[1]
    assert len(flat) == len(set(flat)) == 2 * ROUNDS * engines[0].groups
    # every trained group is within max_policy_age of the merged version
    for r in rollouts:
        for rollout_id, _, versions in r.taken:
            assert all(rollout_id - v <= MAX_POLICY_AGE for v in versions)
    # 4. R OuterSteps
    kinds = [json.loads(line).get("kind") for line in (out / "syncer-tape.jsonl").read_text().splitlines()]
    assert kinds.count("outer_step") == ROUNDS
