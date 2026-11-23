"""A decoupled synced group end to end on CPU: two decoupled trainer islands
(``DecoupledSync`` over the real ``DecoupledRlBridge``), the real Rust syncer
(the launcher's asynchronous decoupled options: quorum 1, a grace window,
tokens2-over-steps, outer Nesterov), and two samplers that follow their own
trainer (``--rl-sampler-follow trainer``).

Each trainer is a fake ports island (``yeto.rl.engine.fake``) wired with the
production publisher wrappers (``sampler_island.wire_policy_releases`` on its
own lane) and the production reader (``buffer_rollout.BatchReader`` over the
whole buffer of its sub-store ``trainer<k>``). Each sampler is a thread over
that sub-store: ``PolicyFollower`` loads each release of its trainer and writes
groups stamped with the version it serves, until its trainer's end record
(``sampler_resume.stop_when_trainers_ended``).

Checks:
1. the two sub-stores are disjoint: each holds only its own trainer's releases
   (``origin.learner_id``) and its own sampler's groups;
2. each sampler loads its trainer's local versions in order, with the hashes
   the trainer published (its local policies, not a merged one);
3. every trained group is within ``max_policy_age`` of the trainer's local
   version, and no buffer position is trained twice;
4. the release records and the tape carry the full fragment version vector of
   each local version, and its merged version (``merged_version``);
5. the syncer ran all T fragment steps and both trainers applied the same final
   cut (the run ends cleanly, the samplers stop on their trainer's end).

The syncer binary is YETO_TEST_SYNCER when set, else a cargo build of syncer/
(as test_rl_decoupled_pool_e2e.py)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_island_pool_e2e import _free_port
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.sampler_island import mark_trainer_end, wire_policy_releases
from yeto.rl.decoupled import DecoupledBridgeConfig
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import DecoupledIslandProgress, DecoupledSync
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import LORA_CONFIG_HASH, MODEL_REVISION, FakeEngine, fake_capabilities
from yeto.rl.engine.full_parameter_sync import merged_version
from yeto.rl.engine.merged_release import OWN_LANE
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sample_roles import SampleBufferConfig, trainer_substore
from yeto.rl.engine.sampler_resume import stop_when_trainers_ended
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

FRAGMENTS, PIPELINE, HORIZON, SWEEPS = 2, 2, 2, 3
TOTAL = FRAGMENTS * SWEEPS
MAX_POLICY_AGE = 2
_EXECUTOR_LOCK = threading.Lock()
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def syncer_binary():
    if os.environ.get("YETO_TEST_SYNCER"):
        return os.environ["YETO_TEST_SYNCER"]
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return str(ROOT / "syncer/target/debug/yeto-syncer")


def _tensors(offset: float) -> dict[str, torch.Tensor]:
    return {
        "base_model.model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]) + offset,
        "base_model.model.b.lora_A.weight": torch.tensor([[3.0, 4.0]]) + offset,
        "base_model.model.c.lora_A.weight": torch.tensor([[5.0]]) + offset,
        "base_model.model.d.lora_A.weight": torch.tensor([[6.0]]) + offset,
    }


def _args(out: Path, k: int, port: int, initial) -> SimpleNamespace:
    """The Miles args a decoupled synced trainer island carries (the fields
    ``DecoupledSync`` and its progress file read)."""
    return SimpleNamespace(
        actor_num_gpus_per_node=1, actor_num_nodes=1, advantage_estimator="grpo",
        yeto_rl_model="org/model", yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION, yeto_rl_data_revision="c" * 40,
        expert_model_parallel_size=1, yeto_rl_layout_hash=initial.layout_hash, lr=1e-5,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH, n_samples_per_prompt=2,
        num_steps_per_rollout=1, pipeline_model_parallel_size=1,
        over_sampling_batch_size=2, yeto_rl_reward_sha256="d" * 64,
        yeto_rl_source_sha256="f" * 64, rollout_batch_size=1, seq_length=128, seed=7,
        rollout_max_response_len=32, custom_generate_function_path=None,
        use_session_server=False, tito_model=None, yeto_rl_learner_id=k,
        yeto_rl_sync_preset="decoupled", yeto_rl_num_fragments=FRAGMENTS,
        yeto_rl_pipeline=PIPELINE, yeto_rl_local_horizon=HORIZON,
        yeto_rl_total_sweeps=SWEEPS, yeto_rl_total_fragment_steps=TOTAL,
        yeto_rl_sync_layout_fingerprint="e" * 64,
        yeto_rl_bridge_config=DecoupledBridgeConfig(
            syncer_addr=("127.0.0.1", port), learner_id=k, total_fragment_steps=TOTAL,
            num_fragments=FRAGMENTS, pipeline=PIPELINE, local_horizon=HORIZON,
            expected_specs=initial.specs, base_model_revision=MODEL_REVISION,
            lora_config_hash=LORA_CONFIG_HASH, canonical_layout_hash=initial.layout_hash,
            wan_streams=0),
        yeto_rl_completed_groups_path=str(out / f"island-{k}.pt"),
        yeto_rl_event_tape=str(out / f"events-{k}.jsonl"),
        # a synced trainer whose samplers follow it (island_entry.apply_sample_role)
        yeto_rl_sample_lane=OWN_LANE,
        yeto_rl_sample_buffer=SampleBufferConfig(role="trainer", store=str(out / trainer_substore(k))),
        lora_rank=16,
    )


class _OwnLaneRollout:
    """The fake rollout pool of one trainer: each round first takes its groups
    from the whole buffer of its sub-store (as ``BufferReaderRolloutFn``), then
    lets the fake pool build the batch. ``_in_executor`` runs a function as in
    the Miles rollout executor, with this island's reader active."""

    def __init__(self, pool, reader: br.BatchReader, groups: int) -> None:
        self.pool, self.reader, self.groups = pool, reader, groups
        self.taken: list[tuple[int, list[int], dict[int, int]]] = []

    def generate(self, rollout_id, **kwargs):
        with _EXECUTOR_LOCK:
            br._ACTIVE[br.TRAINER] = self.reader
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
    """``actor_model.run_plugin(WRITE_LORA_ADAPTER)``: the adapter the engines serve."""

    def __init__(self, engine) -> None:
        self.engine = engine

    async def run_plugin(self, name: str, kwargs: dict) -> None:
        torch.save(self.engine.published_tensors, Path(kwargs["path"]) / "adapter.pt")


def _sampler(store_root: Path, k: int, stop: threading.Event, loads: list, written: list,
             groups_ahead: int) -> None:
    """Sampler of trainer ``k``: loads its releases, writes groups stamped with the
    served version, and stops when its trainer ended (or the test gives up)."""
    store = CheckpointStore(store_root)
    buffer = FileSampleBuffer(store, batch_size=1)
    ended = stop_when_trainers_ended(store, 1)
    follower = PolicyFollower(
        PolicyReleaseStore(store),
        lambda release, directory: loads.append(
            (release.version, release.policy_hash, release.origin,
             {n: t.tolist() for n, t in torch.load(directory / "adapter.pt").items()})))
    n = 0
    while not stop.is_set() and not ended():
        follower.refresh()
        release = follower.release
        if release is None or len(buffer) - int(release.buffer_selected or 0) >= groups_ahead:
            time.sleep(0.005)
            continue
        group_id = f"t{k}-g{n}"
        ref = buffer.write_payload(group_id, br.encode_group([{"tokens": [n]}]))
        buffer.append(SampleIndexEntry(group_id, follower.version, ref))
        written.append((group_id, follower.version))
        n += 1
    buffer.mark_data_end()


def _events(path: Path, kind: str) -> list[dict]:
    return [r for r in (json.loads(line) for line in path.read_text().splitlines())
            if r.get("event") == kind]


def test_decoupled_group_trainers_feed_their_own_samplers_to_the_final_cut(syncer_binary, tmp_path,
                                                                          monkeypatch):
    port, key = _free_port(), "b1-decoupled-group"
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", key)
    out = Path(os.environ.get("YETO_TEST_DECOUPLED_GROUP_OUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("*.jsonl"), *out.glob("island-*.pt"), *out.glob("*.ckpt*")]:
        old.unlink()
    for k in range(2):
        shutil.rmtree(out / trainer_substore(k), ignore_errors=True)
        (out / trainer_substore(k)).mkdir()
    # the launcher's asynchronous decoupled syncer command (yeto.launch.syncer), with a
    # short grace window
    syncer = subprocess.Popen(
        [syncer_binary, "--port", str(port), "--learners", "2", "--quorum", "1",
         "--grace-ms", "2000", "--grace-gamma", "0.8", "--grace-tau", "2.0",
         "--pipeline", str(PIPELINE), "--sync-interval-steps", str(float(HORIZON)),
         "--delta-correction", "none", "--total-steps", str(TOTAL),
         "--outer-lr", "0.7", "--outer-momentum", "0.9", "--max-base-lag", str(FRAGMENTS),
         "--learner-weight", "tokens2-over-steps",
         "--checkpoint-path", str(out / "syncer.ckpt"), "--checkpoint-every", "1",
         "--event-tape", str(out / "syncer-tape.jsonl")],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT)
    stop = threading.Event()
    loads: list[list] = [[], []]
    written: list[list] = [[], []]
    try:
        time.sleep(1.0)
        engines = (FakeEngine(tensors=_tensors(0.0), step_delta=0.25),
                   FakeEngine(tensors=_tensors(0.0), step_delta=0.5))
        samplers = [threading.Thread(target=_sampler, daemon=True,
                                     args=(out / trainer_substore(k), k, stop, loads[k], written[k],
                                           4 * engines[k].groups))
                    for k in range(2)]
        for thread in samplers:
            thread.start()
        drivers, rollouts, args_of = [], [], []
        for k, engine in enumerate(engines):
            args = _args(out, k, port, engine.canonical(0))
            sync = DecoupledSync(args)
            driver = IslandDriver(
                learner_id=k, rollout=engine.rollout, trainer=engine.trainer,
                policy_state=engine.policy_state, publisher=engine.publisher,
                placement=engine.placement, algorithm=AlgorithmSpec(), sync=sync,
                events=EventTape(out / f"events-{k}.jsonl", k),
                capabilities=fake_capabilities("miles-like"),
                progress=DecoupledIslandProgress(args), max_rollouts=200)
            buffer = FileSampleBuffer(CheckpointStore(out / trainer_substore(k)),
                                      batch_size=engine.groups, lane=OWN_LANE)
            reader = br.BatchReader(buffer, max_policy_age=MAX_POLICY_AGE, wait_s=30.0, poll_s=0.01,
                                    cursor_log=br.open_lane_log(buffer))
            rollout = _OwnLaneRollout(driver.rollout, reader, engine.groups)
            driver.rollout = rollout
            wire_policy_releases(driver, actor_model=_Actor(engine),
                                 runner=SimpleNamespace(run=asyncio.run), miles_args=args)
            drivers.append(driver)
            rollouts.append(rollout)
            args_of.append(args)
        results, errors = {}, {}

        def run(k, driver):
            try:
                results[k] = driver.run()
                # the island's normal end (entry): its samplers stop on this record
                assert mark_trainer_end(args_of[k], learner_id=k)
            except BaseException as error:  # noqa: BLE001
                errors[k] = error

        threads = [threading.Thread(target=run, args=(k, d), daemon=True) for k, d in enumerate(drivers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
            assert not thread.is_alive()
        assert errors == {}
        for thread in samplers:  # each sampler stops on its own trainer's end record
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert syncer.wait(15) == 0
    finally:
        stop.set()
        if syncer.poll() is None:
            syncer.kill()

    for k in range(2):
        events = out / f"events-{k}.jsonl"
        published = {e["policy_version"]: e["rl/policy_token"].rsplit(":", 1)[1]
                     for e in _events(events, "rl_publication")}
        released = _events(events, "rl_policy_released")
        # 1. disjoint sub-stores: only this trainer's releases and this sampler's groups
        records = PolicyReleaseStore(CheckpointStore(out / trainer_substore(k))).records()
        assert records and {r.origin["learner_id"] for r in records} == {0}
        assert {r.origin["kind"] for r in records} == {"local"}
        assert all(e["origin"]["num_learners"] == 1 for e in released)
        index = FileSampleBuffer(CheckpointStore(out / trainer_substore(k)), batch_size=1)
        # every group in this sub-store is one its own sampler wrote
        assert len(index) == len(written[k]) > 0
        assert all(group_id.startswith(f"t{k}-") for group_id, _ in written[k])
        # 2. the sampler loads its own trainer's local versions, in order, with their hashes
        versions = [v for v, *_ in loads[k]]
        assert versions == sorted(versions) and versions[0] == 0
        # (the final release lands as the trainer ends: its sampler may stop first)
        order = [e["policy_version"] for e in released]
        assert versions == order[:len(versions)] and len(versions) >= len(order) - 1
        assert all(e["policy_version"] == e["local_version"] for e in released)
        assert all(h == published[v] for v, h, *_ in loads[k])
        assert released[-1]["policy_hash"] == published[max(published)]
        # 3. trained groups within the age window of the local version; none twice
        positions = [p for _, ps, _ in rollouts[k].taken for p in ps]
        assert positions and len(positions) == len(set(positions))
        for rollout_id, _, behavior in rollouts[k].taken:
            assert all(0 <= rollout_id - v <= MAX_POLICY_AGE for v in behavior)
        # 4. the fragment vector and its merged version on every release and publication
        for e in released:
            vector = e["origin"]["fragment_versions"]
            assert len(vector) == FRAGMENTS and e["origin"]["merged_version"] == merged_version(vector)
        snapshots = {e["rl/rollout_id"]: e["rl/fragment_versions"]
                     for e in _events(events, "rl_policy_snapshot")}
        for e in _events(events, "rl_publication"):
            assert e["sync/outer_version"] == merged_version(snapshots[e["policy_version"]])
        final = [e for e in released if e["policy_version"] == max(published)]
        assert final and final[0]["origin"]["fragment_versions"] == [TOTAL - 1, TOTAL]
    # 5. T fragment steps merged; both trainers end on the same final cut
    tape = [json.loads(line) for line in (out / "syncer-tape.jsonl").read_text().splitlines()]
    steps = [r for r in tape if "fragment" in r and "step" in r]
    assert [r["step"] for r in steps] == list(range(1, TOTAL + 1))
    assert [r["fragment"] for r in steps] == [s % FRAGMENTS for s in range(TOTAL)]
    assert set().union(*(r["responded"] for r in steps)) == {0, 1}
    for name in results[0].tensors:
        assert torch.equal(results[0].tensors[name], results[1].tensors[name])
    # the two trainers trained different local paths (independent local versions)
    assert [e["policy_version"] for e in _events(out / "events-0.jsonl", "rl_publication")] != [] \
        and loads[0][1][1] != loads[1][1][1]
