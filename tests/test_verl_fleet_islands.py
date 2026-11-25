"""verl trainer islands of an island fleet: the adapter side.

* per-learner releases (``--rl-release-mode per_learner``): each trainer releases its own
  local versions into its sub-store ``trainer<k>/policy-releases/`` with its own
  ``LATEST.json``, every ``--rl-release-every`` local versions, the last skipped one at the
  end of the run; the synced group's default (learner 0 releases the merged version) is
  unchanged;
* the deterministic stop: the verl training process ends a trainer breaker's stop with the
  shared ``config_errors.island_stop_exit`` (exit 11), a killed worker as before (exit 1);
* the prompt row width of a store buffer trainer: its pools' prompt bound or its sequence
  budget, refused when the rows exceed ``--seq-len``; 512 everywhere else, as before.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
import torch

from test_rl_object_store import FakeS3
from test_rl_verl_buffer_trainer import NAMES, SHAPE, FakeBufferActor, _entry_args, _sample, _State
from yeto.cloud import s3_store
from yeto.config_errors import DETERMINISTIC_STOP_ENV, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER
from yeto.launch import island_lifecycle as lc
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl.buffer_ports import (MERGED_RELEASE, PER_LEARNER_RELEASE, RELEASE_LOG,
                                                BufferReleasePublisher, install_buffer_ports)
from yeto.rl.engine import batch_reader
from yeto.rl.engine.buffer_payload import GroupWriter, SampleBufferRolloutError
from yeto.rl.engine.merged_release import OWN_LANE, BufferLane, ResumedReleases, lane_cursor_log
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore, store_for
from yeto.rl.engine.sample_buffer import FileSampleBuffer

NONZERO = {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.ones(4, 2)}
ZERO = {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.zeros(4, 2)}


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------------------------------------------------------------------- 2.1 per-learner releases
MOUNT = "~/yeto-checkpoint-store/runs/a"


def _substore(tmp_path, s3, learner, stage="stage"):
    """Trainer ``learner``'s sub-store ``trainer<k>`` of one S3 run store (FakeS3), staged on
    local disk and reached by the object API (a one-node island); ``stage``: whose disk."""
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: "s3://bkt/runs/a",
           s3_store.STAGE_ENV: str(tmp_path / stage)}
    return store_for(f"{MOUNT}/trainer{learner}", environ=env, client=s3)


def _fleet_trainer(tmp_path, store, learner, events, *, release_every=1, mode=PER_LEARNER_RELEASE):
    """A synced buffer trainer reading its own lane of ``store`` (an island fleet's trainer)."""
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland

    actor = FakeBufferActor()
    out = tmp_path / f"out{learner}"
    out.mkdir(exist_ok=True)
    island = FullyAsyncIsland(actor, learner_id=learner, limit=2, required=1, out_dir=str(out),
                              readback_dir=str(tmp_path / "rb"))
    island.emit = lambda event, **f: events.append((event, f))
    buffer = FileSampleBuffer(store, batch_size=1, lane=OWN_LANE)
    _, publisher, _ = install_buffer_ports(
        island, buffer=buffer, releases=PolicyReleaseStore(store), shape=SHAPE, base_dir=tmp_path,
        cursor_log=lane_cursor_log(buffer), release_mode=mode, release_every=release_every)

    def merge(base, tensors, directory, _learner=learner):
        # each learner's own weights: the release files say whose they are
        (directory / "model.safetensors").write_bytes(f"learner{_learner}".encode())
        return {}

    publisher.merge = merge
    driver = type("D", (), {"outer_version": None})()
    publisher.bind(driver)
    return actor, island, publisher, driver


def _run_local_versions(trainers, versions, *, outer_of=lambda local: 0):
    for local in versions:
        for learner, (actor, _island, publisher, driver) in trainers.items():
            actor.param_version = local
            driver.outer_version = outer_of(local)
            publisher.publish(_State(local, ZERO if local == 0 else NONZERO, f"l{learner}-v{local}"))


def test_two_fleet_trainers_release_their_own_versions_into_their_own_substores(tmp_path):
    s3 = FakeS3()
    events = {0: [], 1: []}
    stores = {k: _substore(tmp_path, s3, k) for k in (0, 1)}
    trainers = {k: _fleet_trainer(tmp_path, stores[k], k, events[k], release_every=2) for k in (0, 1)}
    # local versions 0..5; the syncer's merges (outer versions) are not what is released
    _run_local_versions(trainers, range(6), outer_of=lambda local: local // 3)
    for _actor, _island, publisher, _driver in trainers.values():
        publisher.drain_releases()  # the end of the run: v5 (skipped by the cadence) goes out

    for k in (0, 1):
        released = [f for e, f in events[k] if e == "rl_policy_released"]
        # v0 is the base (not released), v2 and v4 by the cadence, v5 at the drain
        assert [(f["policy_version"], f["local_version"]) for f in released] == [(2, 2), (4, 4), (5, 5)]
        assert all(f["origin"] == {"kind": "local", "learner_id": 0, "num_learners": 1,
                                   "local_version": f["local_version"]} for f in released)
        skipped = [(f["policy_version"], f["reason"]) for e, f in events[k] if e == "rl_policy_release_skipped"]
        assert skipped == [(1, "release_every"), (3, "release_every"), (5, "release_every")]
        assert ("rl_policy_release_role", {"learner_id": 0, "num_learners": 1, "publisher": True,
                                           "own_lane": True}) in events[k]
        # a reader on another machine (its own stage) sees this trainer's LATEST by the object API
        latest = PolicyReleaseStore(_substore(tmp_path, s3, k, stage="reader")).latest()
        assert (latest.seq, latest.version, latest.weights) == (3, 5, "full")
        assert latest.policy_hash == _State(5, {}, f"l{k}-v5").policy_tensor_hash()
        log = _lines(tmp_path / f"out{k}" / RELEASE_LOG.format(learner=k))
        assert [(r["policy_version"], r["released"]) for r in log] == [
            (0, False), (1, False), (2, True), (3, False), (4, True), (5, False), (5, True)]
        assert log[-1]["drained"] is True and log[-1]["release_version"] == 5
        assert "merged_versions" not in log[-1]  # its groups carry local versions
    # the two trainers' releases never share a key: each has its own release directory and LATEST
    keys = sorted(key for _bucket, key in s3.objects if "/policy-releases/" in key)
    for k in (0, 1):
        own = [key for key in keys if key.startswith(f"runs/a/trainer{k}/policy-releases/")]
        assert f"runs/a/trainer{k}/policy-releases/LATEST.json" in own
        assert sorted({key.split("/")[4] for key in own if key.count("/") > 4}) == ["r2-v4", "r3-v5"]
        weights = [key for key in own if key.endswith("model.safetensors")]
        assert weights and all(s3.objects[("bkt", key)] == f"learner{k}".encode() for key in weights)
    assert all(key.startswith(("runs/a/trainer0/", "runs/a/trainer1/")) for key in keys)


def test_a_fleet_trainer_releases_every_local_version_by_default_and_the_guard_keeps_them_increasing(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    events = []
    trainers = {0: _fleet_trainer(tmp_path, store, 0, events)}
    _run_local_versions(trainers, range(4))
    _run_local_versions(trainers, [2])  # a local version at or below LATEST: never released again
    released = [f["policy_version"] for e, f in events if e == "rl_policy_released"]
    assert released == [1, 2, 3]
    assert [f["reason"] for e, f in events if e == "rl_policy_release_skipped"] == ["not_newer_than_latest"]
    assert not [e for e, _ in events if e == "rl_policy_release_gap"]
    trainers[0][2].drain_releases()  # nothing skipped by a cadence: the drain releases nothing
    assert [f["policy_version"] for e, f in events if e == "rl_policy_released"] == [1, 2, 3]


def test_a_fleet_trainer_judges_its_pools_groups_by_its_own_local_versions(tmp_path):
    """Its pools load its local releases, so a group's behavior version is a local version;
    a synced group's trainer (merged mode) reads the same stamp as a merge it never applied."""
    for mode, expected in ((PER_LEARNER_RELEASE, (1,)), (MERGED_RELEASE, (None,))):
        store = CheckpointStore(tmp_path / mode)
        GroupWriter(FileSampleBuffer(store, batch_size=1), "pool0").write([_sample(1)], group_key="g")
        actor, _island, publisher, driver = _fleet_trainer(tmp_path / mode, store, 0, [], mode=mode)
        for local in (0, 1):  # merged mode: the syncer's outer version stays 0
            actor.param_version, driver.outer_version = local, 0
            publisher.publish(_State(local, ZERO, f"v{local}"))
        assert publisher.feed.pull()["min_steps"] == expected


def _counts_from_zero(rounds):
    """A synced session as the full-parameter one (``FullParameterSync``): its rounds count
    from 0 in every process, and the driver moves the label above the island's releases
    (``continue_version_account_at``); the trainer state stays the one it loaded."""
    from yeto.rl.engine.bridges import LocalOnlySync, _local_state
    from yeto.rl.engine.driver import SyncStart

    class CountsFromZero(LocalOnlySync):
        def continue_version_account_at(self, driver, rollout_id):
            self.resumed_at = int(rollout_id)  # this process still runs ``rounds`` rounds
            return SyncStart(_local_state(driver, int(rollout_id)), int(rollout_id))

    return CountsFromZero(rounds)


def _fleet_process(tmp_path, s3, name, *, rounds, actor, wait_s=30.0, mode=PER_LEARNER_RELEASE):
    """One process of fleet trainer 0 (its sub-store ``trainer0`` on FakeS3, its own lane),
    the real driver over the real ports, and its pool: after each publication the pool
    loads the sub-store's LATEST and writes one group with it (v0 before any release)."""
    from test_rl_verl_buffer_trainer import TOLERANT, _base
    from yeto.rl.adapters.verl import fully_async_round as far
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland, FullyAsyncTrainerGroup
    from yeto.rl.adapters.verl.ports_impl import VerlPolicyState
    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
    from yeto.rl.engine.driver import EventTape, IslandDriver

    store = _substore(tmp_path, s3, 0, stage=f"stage-{name}")  # a migrated process: another disk
    base = tmp_path / "base"
    if not base.is_dir():
        _base(tmp_path, dtype=torch.float32)
    specs = (CanonicalTensorSpec(NAMES[0], (2, 4), "float32", 8), CanonicalTensorSpec(NAMES[1], (4, 2), "float32", 8))
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=1, out_dir=str(tmp_path / name),
                              readback_dir=str(tmp_path / f"rb-{name}"), base_model_revision="a" * 40,
                              lora_config_hash="b" * 64, layout_hash=canonical_layout_hash(specs),
                              expected_specs=specs)
    tape = EventTape(tmp_path / f"{name}.jsonl", 0)
    island.emit = lambda event, **f: tape.append({"event": event, **f})
    buffer = FileSampleBuffer(store, batch_size=1, lane=OWN_LANE, create_index=False)
    releases = PolicyReleaseStore(store)
    rollout, publisher, placement = install_buffer_ports(
        island, buffer=buffer, releases=releases, shape=bf.RowShape(64, 16, 0), base_dir=base,
        tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=wait_s, cursor_log=lane_cursor_log(buffer),
        release_mode=mode)
    pool = GroupWriter(FileSampleBuffer(store, batch_size=1), f"pool-{name}")
    publish = publisher.publish

    def publish_then_pool_follows(state):
        result = publish(state)
        latest = releases.latest()
        version = 0 if latest is None else int(latest.version)
        pool.write([_sample(version, index=0), _sample(version, index=1)],
                   group_key=f"{name}-{state.policy_version}")
        return result

    publisher.publish = publish_then_pool_follows
    profile = far.execution_profile_for(TOLERANT, 1, groups_per_round=1, samples_per_group=2, sync="none")
    driver = IslandDriver(
        learner_id=0, rollout=rollout, trainer=FullyAsyncTrainerGroup(island), policy_state=VerlPolicyState(island),
        publisher=publisher, placement=placement,
        capabilities=verl_capabilities("sha256:" + "0" * 64, fully_async_limit=1),
        algorithm=TOLERANT, sync=_counts_from_zero(rounds), events=tape, profile=profile)
    publisher.bind(driver)
    return driver, island, releases


def test_a_migrated_per_learner_trainer_continues_its_versions_above_its_own_latest(tmp_path):
    """Fleet trainer 0 released v1..v5 into its sub-store, then moved to another cloud (a new
    process, same learner id, a fresh account at 0 from its sync session). Its first
    publication -- the weights it holds after the syncer's catch-up -- is v6, its pool's
    groups of v5 and v6 are trained, and nothing is skipped as not newer than LATEST."""
    s3 = FakeS3()
    first, _island, releases = _fleet_process(tmp_path, s3, "first", rounds=5, actor=FakeBufferActor())
    assert first.run().policy_version == 5
    assert [r.version for r in releases.records()] == [4, 5]  # the store keeps the newest two

    caught_up = FakeBufferActor()  # the weights the syncer's catch-up loaded: not the base
    caught_up.tensors[NAMES[1]] += 0.5
    loaded = {k: v.clone() for k, v in caught_up.tensors.items()}
    driver, island, releases = _fleet_process(tmp_path, s3, "second", rounds=2, actor=caught_up)
    state = driver.run()

    assert state.policy_version == 8 and driver.rounds_completed == 2
    tape = _lines(tmp_path / "second.jsonl")
    [own] = [e for e in tape if e["event"] == "rl_buffer_own_lane_versions"]
    assert (own["latest_version"], own["start_version"], own["known_versions"], own["restored_versions"]) == (
        5, 6, [4, 5], [5])
    [continued] = [e for e in tape if e["event"] == "rl_version_account_continued"]
    assert (continued["rollout_id"], continued["session_start"], continued["latest_release_version"]) == (6, 0, 5)
    assert [e["versions"] for e in tape if e["event"] == "rl_published_policies_restored"] == [[5]]
    assert not [e for e in tape if e["event"] == "rl_policy_release_skipped"]
    released = [e for e in tape if e["event"] == "rl_policy_released"]
    assert [(e["policy_version"], e["local_version"]) for e in released] == [(6, 6), (7, 7), (8, 8)]
    # v6 is the caught-up weights under the next number (only the label moved)
    log = _lines(tmp_path / "second" / RELEASE_LOG.format(learner=0))
    assert log[0]["policy_version"] == 6 and log[0]["released"] is True
    assert log[0]["policy_hash"] == released[0]["policy_hash"] == island.hashes[6]
    from yeto.rl.adapters.verl.merged_release import write_merged_checkpoint

    expected = write_merged_checkpoint(tmp_path / "base", loaded, tmp_path / "expected-v6")
    assert log[0]["merge"] == expected and expected["max_relative_change"] > 0
    # the pool kept up: the groups of v5 (loaded before the move) and v6 are trained
    consumed = _lines(tmp_path / "second" / bf_consumed(0))
    assert [(r["behavior_versions"], r["kept"]) for r in consumed] == [([5, 5], True), ([6, 6], True)]
    assert releases.latest().version == 8
    assert island.consume_map.to_dict() == {str(v): v for v in range(4, 9)}


def bf_consumed(learner):
    from yeto.rl.adapters.verl.buffer_ports import CONSUMED_LOG

    return CONSUMED_LOG.format(learner=learner)


def test_a_merged_mode_or_first_start_trainer_does_not_continue_any_account(tmp_path):
    """Only a per-learner trainer reads its own releases: a synced group's trainer (merged
    mode) and a per-learner trainer whose sub-store is empty start as before."""
    store = CheckpointStore(tmp_path / "store")
    releases = PolicyReleaseStore(store)
    for version in (1, 2):
        releases.release(version, f"{version:064d}", lambda d: (d / "w.bin").write_text("w"))
    for mode, store_dir in ((MERGED_RELEASE, "store"), (PER_LEARNER_RELEASE, "empty")):
        events = []
        (tmp_path / mode).mkdir()
        _, island, publisher, driver = _fleet_trainer(tmp_path / mode, CheckpointStore(tmp_path / store_dir),
                                                      0, events, mode=mode)
        assert publisher.own_releases is None
        assert getattr(driver, "published_policy_history", None) is None
        assert island.consume_map.to_dict() == {} and island.hashes == {}
        assert not [e for e, _ in events if e == "rl_buffer_own_lane_versions"]
    events = []
    (tmp_path / "own").mkdir()
    _, island, publisher, driver = _fleet_trainer(tmp_path / "own", store, 0, events)
    assert driver.published_policy_history() == [(1, f"{1:064d}"), (2, f"{2:064d}")]
    # max_policy_age 2 below the first publication v3: v1 and v2 name their groups
    assert island.consume_map.to_dict() == {"1": 1, "2": 2}
    assert island.hashes == {1: f"{1:064d}", 2: f"{2:064d}"}
    [(_, own)] = [(e, f) for e, f in events if e == "rl_buffer_own_lane_versions"]
    assert (own["latest_version"], own["start_version"]) == (2, 3)


def test_per_learner_releases_need_a_trainer_on_its_own_lane_and_the_cadence_needs_them(tmp_path):
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland

    store = CheckpointStore(tmp_path / "store")
    island = FullyAsyncIsland(FakeBufferActor(), learner_id=1, limit=2, required=1,
                              out_dir=str(tmp_path / "out"), readback_dir=str(tmp_path / "rb"))
    shared = FileSampleBuffer(store, batch_size=1, lane=BufferLane(1, 2))
    feed = bf.BufferFeed(FileSampleBuffer(store, batch_size=1), vmap=lambda: island.vmap, shape=SHAPE,
                         add_group=print, record=print)
    with pytest.raises(ValueError, match="own lane"):  # a shared lane of a synced group
        install_buffer_ports(island, buffer=shared, releases=PolicyReleaseStore(store), shape=SHAPE,
                             base_dir=tmp_path, cursor_log=lane_cursor_log(shared),
                             release_mode=PER_LEARNER_RELEASE)
    with pytest.raises(ValueError, match="own lane"):  # a single trainer (no lane)
        BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path,
                               release_mode=PER_LEARNER_RELEASE)
    with pytest.raises(ValueError, match="needs --rl-release-mode per_learner"):
        BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path, release_every=2)
    with pytest.raises(ValueError, match="expected one of"):
        BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path, release_mode="local")
    own = ResumedReleases(3, 2, ((2, "2" * 64), (3, "3" * 64)))
    with pytest.raises(ValueError, match="only a per-learner trainer continues"):  # merged mode
        BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path,
                               lane=BufferLane(1, 2), own_releases=own)
    with pytest.raises(ValueError, match="only a per-learner trainer continues"):  # a single trainer
        BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path, own_releases=own)


FLEET_TRAINER = ("--rl-sample-role", "trainer", "--syncer", "h:1", "--rl-max-policy-age", "2",
                 "--rl-sample-store", "/s/trainer1", "--rl-sample-lanes", "1", "--learner-id", "1")


def test_island_entry_plans_per_learner_releases_and_keeps_the_default_plan_unchanged():
    from yeto.rl.adapters.verl.island_entry import sample_buffer_section

    base = {"store": "/s/trainer1", "wait_s": None, "lane": "0/1", "release_delta_full_every": 0}
    # the default and an explicit merged mode: the plan section of main, key for key
    assert sample_buffer_section(_entry_args(*FLEET_TRAINER)) == base
    assert sample_buffer_section(_entry_args(*FLEET_TRAINER, "--rl-release-mode", "merged")) == base
    assert sample_buffer_section(_entry_args(*FLEET_TRAINER, "--rl-release-mode", "per_learner")) == {
        **base, "release_mode": "per_learner"}
    assert sample_buffer_section(_entry_args(*FLEET_TRAINER, "--rl-release-mode", "per_learner",
                                             "--rl-release-every", "2")) == {
        **base, "release_mode": "per_learner", "release_every": 2}


@pytest.mark.parametrize(("extra", "message"), [
    # a synced group's shared lane, and a single trainer: no own sub-store
    (("--rl-release-mode", "per_learner", "--rl-sample-lanes", "2"), "own sub-store"),
    (("--rl-release-every", "2"), "needs --rl-release-mode per_learner"),
])
def test_island_entry_refuses_per_learner_releases_off_its_own_lane(extra, message):
    from yeto.rl.adapters.verl.island_entry import sample_buffer_section

    with pytest.raises(SystemExit, match=message):
        sample_buffer_section(_entry_args(*FLEET_TRAINER, *extra))
    with pytest.raises(SystemExit, match="own sub-store"):
        sample_buffer_section(_entry_args("--rl-sample-role", "trainer", "--rl-single-island-no-sync",
                                          "--rl-max-policy-age", "1", "--rl-resume-store", "/s",
                                          "--rl-release-mode", "per_learner"))


def test_the_verl_backend_declares_per_learner_releases():
    from yeto.rl.adapters.verl import identity
    from yeto.rl.engine import backend_identity

    assert backend_identity.PER_LEARNER_RELEASE == "per_learner_release"
    assert backend_identity.FLEET_FULL_PARAMETER == "fleet_full_parameter"
    assert backend_identity.FLEET_PARTIAL_EXPERTS == "fleet_partial_experts"
    assert backend_identity.PER_LEARNER_RELEASE in identity.CAPABILITIES


# ---------------------------------------------------------------------- 3.1 deterministic stop
@pytest.fixture
def verl_main():
    from verl_island_fakes import modules, stub_verl_and_ray

    with stub_verl_and_ray():
        (module,) = modules("yeto.rl.adapters.verl.verl_main")
        yield module


def _sample_wait():
    error = SampleBufferRolloutError("rollout 3: 0/4 groups after 300 s without a new group")
    batch_reader.trip_sample_wait(None, waited_s=301.0, limit_s=300.0, error=error)
    return error


class RayTaskError(Exception):
    """A Ray task error as ``ray.get`` raises it: the remote error is its ``cause``."""

    def __init__(self, cause):
        super().__init__(f"ray::YetoBufferTaskRunner.run() {type(cause).__name__}: {cause}")
        self.cause = cause


class RayActorError(Exception):
    """A Ray worker killed from outside: no result, no breaker."""


def _raise(error):
    def run():
        raise error
    return run


def test_a_verl_island_ends_a_sample_wait_stop_with_the_island_stop_exit(verl_main, monkeypatch, capsys):
    monkeypatch.setenv(DETERMINISTIC_STOP_ENV, "1")
    for error in (_sample_wait(), RayTaskError(_sample_wait())):
        with pytest.raises(SystemExit) as stopped:
            verl_main.run_island(_raise(error))
        assert stopped.value.code == ISLAND_STOP_EXIT == 11
        err = capsys.readouterr().err
        assert f"{ISLAND_STOP_MARKER} {lc.BREAKER} trainer.sample_wait" in err
        # the launcher reads the exit and the line as a stop, never relaunched
        found = lc.classify(exit_code=stopped.value.code, log_text=err)
        assert (found.cause, found.relaunch) == (lc.BREAKER, False)
    assert verl_main.run_island(lambda: {"rounds": 3}) == {"rounds": 3}


def test_a_killed_verl_worker_stays_an_infrastructure_failure(verl_main, monkeypatch, capsys):
    monkeypatch.setenv(DETERMINISTIC_STOP_ENV, "1")
    for error in (RayActorError("The actor died unexpectedly before finishing this task"),
                  RayTaskError(RuntimeError("NCCL error: unhandled system error"))):
        with pytest.raises(type(error)):  # raised: hydra exits 1, the island process exits 1
            verl_main.run_island(_raise(error))
        assert ISLAND_STOP_MARKER not in capsys.readouterr().err
        found = lc.classify(exit_code=1, log_text="")
        assert (found.cause, found.relaunch) == (lc.INFRA, True)


def test_switched_off_a_verl_island_raises_every_stop_as_before(verl_main, monkeypatch, capsys):
    monkeypatch.delenv(DETERMINISTIC_STOP_ENV, raising=False)
    with pytest.raises(SampleBufferRolloutError):
        verl_main.run_island(_raise(_sample_wait()))
    assert capsys.readouterr().err == ""


def test_every_verl_training_entry_runs_its_island_under_the_stop_exit():
    source = Path(__file__).resolve().parents[1].joinpath("yeto/rl/adapters/verl/verl_main.py").read_text()
    gets = re.findall(r"result = (.*ray\.get\(.*)", source)
    assert len(gets) == 3 and all(g.startswith("run_island(lambda: ray.get(") for g in gets)


def test_the_island_stop_exit_and_its_code_exist_once(monkeypatch):
    """One exit code (``config_errors.ISLAND_STOP_EXIT``) and one exit function: the Miles
    island entry's is the shared one, and no verl module spells the code."""
    from yeto import config_errors
    from yeto.rl.adapters.miles import island_entry as miles_entry

    root = Path(__file__).resolve().parents[1] / "yeto"
    defined = [str(p.relative_to(root)) for p in root.rglob("*.py")
               if re.search(r"^ISLAND_STOP_EXIT\s*=", p.read_text(), re.M)]
    assert defined == ["config_errors.py"]
    verl = root / "rl" / "adapters" / "verl"
    assert not [p.name for p in verl.glob("*.py") if re.search(r"(exit|EXIT)\w*\s*=\s*11\b", p.read_text())]
    monkeypatch.setenv(DETERMINISTIC_STOP_ENV, "1")
    assert miles_entry.island_stop_exit(_sample_wait(), io.StringIO()) == config_errors.island_stop_exit(
        _sample_wait(), io.StringIO()) == ISLAND_STOP_EXIT


# ---------------------------------------------------------------------- 4.2 prompt width
def test_the_prompt_width_comes_from_the_pools_bound_or_the_sequence_budget():
    assert bf.buffer_prompt_width(seq_len=40960, response_len=8192, pool_max_prompt_len=16384) == 16384
    assert bf.buffer_prompt_width(seq_len=40960, response_len=8192) == 32768  # the budget
    assert bf.buffer_prompt_width(seq_len=24576, response_len=8192, pool_max_prompt_len=16384) == 16384  # fits
    with pytest.raises(bf.PromptWidthError, match=r"16384 \+ --rollout-max-response-len 8192 = 24576 tokens per row > the trainer.s --seq-len 24575"):
        bf.buffer_prompt_width(seq_len=24575, response_len=8192, pool_max_prompt_len=16384)
    with pytest.raises(bf.PromptWidthError, match="no prompt tokens"):
        bf.buffer_prompt_width(seq_len=8192, response_len=8192)
    with pytest.raises(bf.PromptWidthError, match="must be positive"):
        bf.buffer_prompt_width(seq_len=8192, response_len=64, pool_max_prompt_len=0)


def _width(*extra, section=True):
    from yeto.rl.adapters.verl.island_entry import island_prompt_width, sample_buffer_section

    args = _entry_args(*extra)
    return island_prompt_width(args, sample_buffer_section(args) if section else None)


SINGLE = ("--rl-sample-role", "trainer", "--rl-single-island-no-sync", "--rl-max-policy-age", "1",
          "--rl-resume-store", "/s")


def test_the_island_prompt_width_of_each_island_kind():
    from yeto.rl.adapters.verl import island_entry

    # island_entry.main sizes the rows (data.max_prompt_length) with this function
    assert "max_prompt = island_prompt_width(args, buffer_section)" in Path(island_entry.__file__).read_text()
    # main's 512: a colocated island, a single buffer trainer, a synced group's trainer
    assert _width(section=False) == 512
    assert _width(*SINGLE) == 512
    assert _width(*FLEET_TRAINER) == 512
    # a store buffer trainer that names its pools' bound, and a fleet trainer (the budget: 512 - 64)
    assert _width(*SINGLE, "--rl-buffer-max-prompt-len", "300") == 300
    assert _width(*FLEET_TRAINER, "--rl-release-mode", "per_learner") == 448
    assert _width(*FLEET_TRAINER, "--rl-release-mode", "per_learner", "--rl-buffer-max-prompt-len", "200") == 200
    # rows past --seq-len: refused before any model or data work
    with pytest.raises(SystemExit, match="raise --seq-len to 1064"):
        _width(*SINGLE, "--rl-buffer-max-prompt-len", "1000")
    # the flag where nothing reads it
    with pytest.raises(SystemExit, match="for a verl buffer trainer reading the store"):
        _width("--rl-buffer-max-prompt-len", "300", section=False)


def test_an_agentic_long_prompt_group_is_trained_on_wide_rows_and_dropped_on_512(tmp_path):
    """The group a 512-wide trainer drops whole (over_length) is kept on the rows the pools'
    bound gives (prompt 2000 tokens, response 3)."""
    long_prompt = tuple(range(1, 2001))
    for width, kept in ((bf.LEGACY_PROMPT_WIDTH, False),
                        (bf.buffer_prompt_width(seq_len=4096, response_len=8, pool_max_prompt_len=2048), True)):
        store = CheckpointStore(tmp_path / str(width))
        writer = GroupWriter(FileSampleBuffer(store, batch_size=1), "pool0")
        writer.write([_sample(0, prompt=long_prompt, index=0), _sample(0, prompt=long_prompt, index=1)],
                     group_key="agentic")
        writer.write([_sample(0, index=0), _sample(0, index=1)], group_key="short")
        from yeto.rl.adapters.verl.fully_async_translate import VersionMap

        vmap, added, records = VersionMap(), [], []
        vmap.record(0, 0)
        feed = bf.BufferFeed(FileSampleBuffer(store, batch_size=1), vmap=lambda: vmap,
                             shape=bf.RowShape(width, 8, 0), add_group=added.append, record=records.append,
                             sleep=lambda s: None)
        meta = feed.pull()
        feed.keep()
        assert (meta["uid"] == "buf0") is kept
        assert (feed.dropped.get(bf.OVER_LENGTH_REASON, 0) == 0) is kept
        if kept:
            assert len(added[0]["tensors"]["prompts"][0]) == 2048
