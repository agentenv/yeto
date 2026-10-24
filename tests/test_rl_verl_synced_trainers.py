"""S25 C3V end to end on CPU: two synced verl buffer trainers, the real Rust
syncer (elastic OuterStep merge), one engine sampler island that follows the
merged releases.

The trainers are the verl buffer trainer of test_rl_verl_buffer_trainer.py
(fake trainer actor; real driver, fully_async rollout port, feed, publisher,
merged checkpoint) over one lane each of one buffer, with ``PoolAvgSync``.
The sampler is the real engine sampler island (fake engine) in a thread.

Checks (the same as the GPU run's, SESSION25/gpu-review/T-C3V.md):
1. both trainers apply the same merged policy hash at every outer version;
2. the sampler loads only versions the publisher released, in order, and each
   released hash is the merged hash of that outer version; learner 1 releases
   nothing;
3. the two lanes are disjoint and no buffer position is trained twice;
4. the syncer ran R OuterSteps.

Runs when YETO_TEST_SYNCER points to a syncer binary (as
test_rl_synced_trainers_sampler.py)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from test_rl_island_pool_e2e import _free_port
from test_rl_verl_buffer_trainer import NAMES, FakeBufferActor, _base
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl.buffer_ports import RELEASE_LOG, install_buffer_ports
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.merged_release import BufferLane, lane_cursor_log
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, open_trainer_buffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import SamplingParams

ROUNDS = 3
GROUPS = 2
MAX_POLICY_AGE = 2
TOLERANT = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=MAX_POLICY_AGE),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def _island(out: Path, store_root: Path, base: Path, learner_id: int, port: int, key: str):
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland, FullyAsyncTrainerGroup
    from yeto.rl.adapters.verl.ports_impl import VerlPolicyState
    from yeto.rl.bridge import BridgeConfig
    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
    from yeto.rl.engine.bridges import PoolAvgSync
    from yeto.rl.engine.driver import EventTape, IslandDriver

    specs = (CanonicalTensorSpec(NAMES[0], (2, 4), "float32", 8),
             CanonicalTensorSpec(NAMES[1], (4, 2), "float32", 8))
    actor = FakeBufferActor()
    island = FullyAsyncIsland(actor, learner_id=learner_id, limit=MAX_POLICY_AGE, required=GROUPS,
                              out_dir=str(out / f"out{learner_id}"), readback_dir=str(out / f"rb{learner_id}"),
                              base_model_revision="a" * 40, lora_config_hash="b" * 64,
                              layout_hash=canonical_layout_hash(specs), expected_specs=specs)
    (out / f"out{learner_id}").mkdir(exist_ok=True)
    tape = EventTape(out / f"tape-{learner_id}.jsonl", learner_id)
    island.emit = lambda event, **f: tape.append({"event": event, **f})
    store = CheckpointStore(store_root)
    buffer = open_trainer_buffer(store, batch_size=1, create_index=False, lane=BufferLane(learner_id, 2))
    rollout, publisher, placement = install_buffer_ports(
        island, buffer=buffer, releases=PolicyReleaseStore(store), shape=bf.RowShape(64, 16, 0),
        base_dir=base, tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=30,
        cursor_log=lane_cursor_log(buffer))
    bridge = BridgeConfig(syncer_addr=("127.0.0.1", port), learner_id=learner_id, global_rounds=ROUNDS,
                          groups_per_round=GROUPS, samples_per_group=2, local_optimizer_steps=1,
                          expected_specs=specs, base_model_revision="a" * 40, lora_config_hash="b" * 64,
                          layout_hash=canonical_layout_hash(specs), event_tape=str(out / f"bridge-{learner_id}.jsonl"),
                          wan_streams=1)
    profile = far.execution_profile_for(TOLERANT, MAX_POLICY_AGE, groups_per_round=GROUPS, samples_per_group=2,
                                        sync="elastic")
    driver = IslandDriver(
        learner_id=learner_id, rollout=rollout, trainer=FullyAsyncTrainerGroup(island),
        policy_state=VerlPolicyState(island), publisher=publisher, placement=placement,
        capabilities=verl_capabilities("sha256:" + "0" * 64, fully_async_limit=MAX_POLICY_AGE),
        algorithm=TOLERANT, sync=PoolAvgSync(bridge, base_wait_s=30.0, syncer_epoch=5, lease_s=6.0),
        events=tape, profile=profile)
    publisher.bind(driver)
    return driver, island


@pytest.mark.skipif(not os.environ.get("YETO_TEST_SYNCER"),
                    reason="needs YETO_TEST_SYNCER (syncer binary)")
def test_two_synced_verl_trainers_feed_one_engine_sampler(tmp_path, monkeypatch):
    port, key = _free_port(), "c3v-e2e"
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", key)  # the islands sign their HELLO
    out = tmp_path
    store_root = out / "store"
    store = CheckpointStore(store_root)
    base, _ = _base(tmp_path, dtype=torch.float32)
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", str(ROUNDS), "--event-tape", str(out / "syncer-tape.jsonl"),
         "--quorum", "1", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-base-lag", "2", "--pipeline", "1",
         "--learner-weight", "tokens2-over-steps", "--island-lease-s", "6", "--syncer-epoch", "5"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT)
    done, sampler_events = threading.Event(), []
    try:
        time.sleep(1.0)
        config = EngineSamplerConfig(sampler_id="s0", num_rollouts=200, max_backlog_groups=4 * GROUPS,
                                     sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32,
                                     poll_s=0.01)
        sampler = build_engine_sampler(
            config, engine=FakeEngine(max_model_len=64), releases=PolicyReleaseStore(store),
            buffer=FileSampleBuffer(store, batch_size=1), emit=lambda n, **f: sampler_events.append((n, f)),
            prompts=lambda r: [PromptItem(f"p{r}", f"prompt {r}", label="x")],
            reward=lambda item, c: float(len(c.token_ids) % 2),
            checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=()),  # N2-fix: no probe pairs
            should_stop=done.is_set, sleep=lambda s: time.sleep(0.005))
        sampler_thread = threading.Thread(target=sampler.run, daemon=True)
        sampler_thread.start()
        islands = [_island(out, store_root, base, i, port, key) for i in range(2)]
        results, errors = {}, {}

        def run(i, driver):
            try:
                results[i] = driver.run()
            except BaseException as error:  # noqa: BLE001
                errors[i] = error

        threads = [threading.Thread(target=run, args=(i, d), daemon=True) for i, (d, _) in enumerate(islands)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=90)
            assert not t.is_alive()
        assert errors == {}
        assert syncer.wait(15) == 0
    finally:
        done.set()
        if syncer.poll() is None:
            syncer.kill()

    tapes = [_lines(out / f"tape-{i}.jsonl") for i in range(2)]
    # 1. the same merged policy on both trainers at every outer version
    applied = [{e["sync/outer_version"]: e["sync/global_policy_hash"] for e in t if e["event"] == "rl_policy_apply"}
               for t in tapes]
    assert sorted(applied[0]) == list(range(ROUNDS + 1)) and applied[0] == applied[1]
    # 2. releases: the publisher only, merged versions increasing, each with the merged hash
    released = [e for e in tapes[0] if e["event"] == "rl_policy_released"]
    assert [e["policy_version"] for e in released] == list(range(1, ROUNDS + 1))  # v0 is the base
    assert all(e["policy_hash"] == applied[0][e["policy_version"]] for e in released)
    assert all(e["origin"]["learner_id"] == 0 for e in released)
    assert not [e for e in tapes[1] if e["event"] == "rl_policy_released"]
    assert not [e for t in tapes for e in t if e["event"] == "rl_policy_release_gap"]
    loaded = [f["policy_version"] for n, f in sampler_events if n == "rl_sampler_policy_loaded"]
    assert loaded == sorted(set(loaded)) and set(loaded) <= {e["policy_version"] for e in released}
    latest = PolicyReleaseStore(store).latest()
    assert latest.version == ROUNDS and latest.origin["learner_id"] == 0
    # 3. disjoint lanes, no position trained twice, lane cursor lines per round
    lanes = [_lines(store_root / "sample-buffer" / "lanes" / f"{i}-of-2.jsonl") for i in range(2)]
    positions = [[p for line in lane for p in line["positions"]] for lane in lanes]
    assert all(p % 2 == i for i, ps in enumerate(positions) for p in ps)
    flat = positions[0] + positions[1]
    assert len(flat) == len(set(flat)) == 2 * ROUNDS * GROUPS
    assert [len(lane) for lane in lanes] == [ROUNDS, ROUNDS]
    # the islands paired merged versions with local ones identically (one merge per round)
    for i in range(2):
        records = _lines(out / f"out{i}" / RELEASE_LOG.format(learner=i))
        assert records[-1]["merged_versions"] == {str(v): v for v in range(ROUNDS + 1)}
    # 4. R OuterSteps
    kinds = [json.loads(line).get("kind") for line in (out / "syncer-tape.jsonl").read_text().splitlines()]
    assert kinds.count("outer_step") == ROUNDS
