"""rl-trainer-start-stop L4.6 (CPU, no Ray, no Miles): sampler / trainer islands over
the durable sample buffer -- sample roles and launch flags, policy releases through
the store, the sampler loop (release loading, backlog cap, trainer stopped), the
engine weight load order, the releasing publisher and the group renumbering."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.engine import sample_roles
from yeto.rl.engine.policy_release import (
    LATEST,
    RELEASE_DIR,
    PolicyFollower,
    PolicyReleaseStore,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def _write_weights(directory: Path) -> None:
    (directory / "model.safetensors").write_text("w")


# ---------------------------------------------------------------- roles and launch
def _launch_args(**kw):
    base = dict(rl_sample_role="sampler", rl_single_island_no_sync=True,
                rl_checkpoint_store="s3://b/run", rl_elastic=False, rl_max_policy_age=0)
    return SimpleNamespace(**{**base, **kw})


def test_role_both_is_the_default_and_adds_nothing():
    args = SimpleNamespace()
    assert sample_roles.sample_role(args) == "both"
    sample_roles.check_launch(args)
    assert sample_roles.island_flags(args) == ""
    assert sample_roles.resolve(args, groups_per_round=4) is None
    with pytest.raises(ValueError, match="needs --rl-sample-role sampler or trainer"):
        sample_roles.check_launch(SimpleNamespace(rl_sample_wait_s=5.0))


@pytest.mark.parametrize("kw, message", [
    ({"rl_single_island_no_sync": False}, "--rl-single-island-no-sync"),
    ({"rl_checkpoint_store": None}, "--rl-checkpoint-store"),
    ({"rl_elastic": True}, "--rl-elastic"),
    ({"rl_sample_role": "trainer"}, "--rl-max-policy-age >= 1"),
    ({"rl_sampler_max_backlog_rounds": 0}, "a value >= 1"),
    ({"rl_sample_wait_s": 3.0}, "--rl-sample-role trainer"),
    ({"rl_sample_role": "x"}, "expected one of"),
])
def test_launch_refuses_unsupported_role_combinations(kw, message):
    with pytest.raises(ValueError, match=message):
        sample_roles.check_launch(_launch_args(**kw))


def test_launch_forwards_the_role_flags_and_the_island_resolves_them():
    from yeto import launcher

    args = _launch_args(rl_sampler_max_backlog_rounds=2)
    sample_roles.check_launch(args)
    _, flags = launcher._ports_infra_flags(args)
    assert " --rl-sample-role sampler --rl-sampler-max-backlog-rounds 2" in flags
    _, default = launcher._ports_infra_flags(SimpleNamespace())
    assert "--rl-sample-role" not in default
    island = SimpleNamespace(rl_sample_role="sampler", rl_resume_store="/mnt/store",
                             rl_sampler_max_backlog_rounds=2)
    assert sample_roles.resolve(island, groups_per_round=8) == sample_roles.SampleBufferConfig(
        role="sampler", store="/mnt/store", max_backlog_groups=16)
    trainer = SimpleNamespace(rl_sample_role="trainer", rl_resume_store="/mnt/store",
                              rl_sample_wait_s=30.0)
    assert sample_roles.resolve(trainer, groups_per_round=8).wait_s == 30.0
    with pytest.raises(ValueError, match="--rl-resume-store"):
        sample_roles.resolve(SimpleNamespace(rl_sample_role="trainer"), groups_per_round=8)


def test_role_flags_exist_on_both_layers():
    import argparse

    from yeto.rl.engine.run_flags import CLI, ISLAND, flags_for

    for layer in (CLI, ISLAND):
        parser = argparse.ArgumentParser()
        for flag in flags_for(layer):
            if flag.options[0] in ("--rl-sample-role", "--rl-sample-wait-s",
                                   "--rl-sampler-max-backlog-rounds"):
                parser.add_argument(*flag.options, **dict(flag.kwargs))
        ns = parser.parse_args(["--rl-sample-role", "trainer", "--rl-sample-wait-s", "7.5"])
        assert (ns.rl_sample_role, ns.rl_sample_wait_s, ns.rl_sampler_max_backlog_rounds) == (
            "trainer", 7.5, None)
        assert parser.parse_args([]).rl_sample_role == "both"


def test_apply_sample_role_sets_the_buffer_config_and_the_island_environment():
    from yeto.rl.adapters.miles.island_entry import apply_sample_role

    miles_args, environ = SimpleNamespace(), {}
    apply_sample_role(SimpleNamespace(groups_per_round=4, learner_id=0), miles_args, environ)
    assert environ == {} and not hasattr(miles_args, "yeto_rl_sample_buffer")
    args = SimpleNamespace(groups_per_round=4, learner_id=3, rl_sample_role="sampler",
                           rl_resume_store="/mnt/store")
    apply_sample_role(args, miles_args, environ)
    assert miles_args.yeto_rl_sample_buffer.max_backlog_groups == 16
    assert environ == {br.BUFFER_ROOT_ENV: "/mnt/store", br.SAMPLER_ID_ENV: "island3"}


def test_ray_workers_get_the_buffer_environment():
    from yeto.rl.adapters.miles.entry import connect_island_ray

    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    env = {br.BUFFER_ROOT_ENV: "/mnt/store", br.MAX_POLICY_AGE_ENV: "2"}
    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379", **env}, ray_module=ray)
    workers = seen["runtime_env"]["env_vars"]
    assert {k: workers[k] for k in env} == env
    assert br.SAMPLER_ID_ENV not in workers


# ---------------------------------------------------------------- Miles argv
def _full_config(role, **kw):
    from tests.test_rl_miles_adapter_config import make_config, sub

    cfg = sub(make_config(), "trainable", parameter_mode="full", target_modules=())
    return dataclasses.replace(cfg, sample_buffer=sample_roles.SampleBufferConfig(
        role=role, store="/mnt/store", **kw))


def _tolerant():
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    return AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))


def test_miles_argv_of_the_sampler_and_the_trainer():
    from tests.test_rl_miles_adapter_config import flag_value, make_config, sub
    from yeto.rl.adapters.miles import config as mc

    base = mc.translate_run_config(sub(make_config(), "trainable", parameter_mode="full",
                                       target_modules=()), _tolerant())
    sampler = mc.translate_run_config(_full_config("sampler", max_backlog_groups=4), _tolerant())
    assert [a for a in sampler.argv if a not in base.argv] == [
        "--rollout-function-path", br.ROLLOUT_FUNCTION_PATHS["sampler"], "--debug-rollout-only"]
    trainer = mc.translate_run_config(_full_config("trainer"), _tolerant(), max_policy_age=1)
    assert flag_value(trainer.argv, "--rollout-function-path") == br.ROLLOUT_FUNCTION_PATHS["trainer"]
    assert flag_value(trainer.argv, "--save-hf") == mc.RELEASE_EXPORTER_SAVE_HF
    assert "--debug-rollout-only" not in trainer.argv and not trainer.rollout_overlap
    # partitioned + policy age: the buffer rollout replaces Miles' fully_async
    partitioned = sub(_full_config("trainer"), "serving", offload_train=False)
    partitioned = dataclasses.replace(partitioned, parallel=dataclasses.replace(
        make_config(colocated=False).parallel))
    assert "--fully-async" not in mc.translate_run_config(partitioned, _tolerant(), max_policy_age=1).argv


def test_miles_argv_refuses_a_lora_buffer_island():
    from tests.test_rl_miles_adapter_config import make_config
    from yeto.rl.adapters.miles import config as mc

    cfg = dataclasses.replace(make_config(), sample_buffer=sample_roles.SampleBufferConfig(
        role="sampler", store="/mnt/store", max_backlog_groups=4))
    with pytest.raises(mc.MilesConfigError, match="full-parameter mode only"):
        mc.translate_run_config(cfg, _tolerant())


# ---------------------------------------------------------------- releases
def test_release_is_committed_by_latest_and_republish_is_a_no_op(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path), keep=2, clock=lambda: 1.0)
    assert releases.latest() is None
    with pytest.raises(RuntimeError, match="export failed"):
        releases.release(1, "h1", lambda d: (_ for _ in ()).throw(RuntimeError("export failed")))
    assert releases.latest() is None  # a failed export never becomes LATEST
    first = releases.release(1, "h1", _write_weights, buffer_selected=4)
    assert (first.seq, first.path, first.buffer_selected) == (1, "r1-v1", 4)
    assert releases.resolve(first) == tmp_path / RELEASE_DIR / "r1-v1"
    assert releases.release(1, "h1", _write_weights) is None  # a restarted trainer republishes
    releases.release(2, "h2", _write_weights)
    rewind = releases.release(1, "h1b", _write_weights)  # restored from an older cut
    assert (rewind.seq, rewind.version) == (3, 1)
    assert sorted(p.name for p in (tmp_path / RELEASE_DIR).iterdir()) == [LATEST, "r2-v2", "r3-v1"]


def test_follower_loads_each_new_release_once_and_keeps_the_version_on_failure(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    loaded, fail = [], [False]

    def load(release, directory):
        if fail[0]:
            raise RuntimeError("engine down")
        loaded.append((release.version, directory.name))

    follower = PolicyFollower(releases, load)
    assert follower.refresh() is None and follower.version == 0
    releases.release(1, "h1", _write_weights)
    assert follower.refresh().version == 1 and follower.refresh() is None
    releases.release(2, "h2", _write_weights)
    fail[0] = True
    with pytest.raises(RuntimeError, match="engine down"):
        follower.refresh()
    assert follower.version == 1  # still serving v1
    fail[0] = False
    follower.refresh()
    assert loaded == [(1, "r1-v1"), (2, "r2-v2")] and follower.version == 2


# ---------------------------------------------------------------- sampler loop
class _Sampler:
    """A fake sampler island: each round writes 2 groups at the served version."""

    def __init__(self, root, releases, max_backlog_groups=4, rounds=6):
        self.events = []
        self.served = []
        self.follower = PolicyFollower(releases, lambda r, d: self.served.append(r.version))
        writer_buffer = FileSampleBuffer(CheckpointStore(root), batch_size=2)
        self.writer = br.GroupWriter(writer_buffer, "island0-a")
        self.sleeps = 0
        self.on_sleep = lambda: None
        self.loop = SamplerLoop(
            follower=self.follower, buffer=FileSampleBuffer(CheckpointStore(root), batch_size=2),
            generate=self.generate, num_rollouts=rounds, max_backlog_groups=max_backlog_groups,
            emit=lambda event, **f: self.events.append((event, f)), poll_s=0.0,
            sleep=self.sleep, should_stop=lambda: self.sleeps > 20)

    def generate(self, rid):
        for g in range(2):  # Miles numbers groups from 0 in every round
            self.writer.write(_group(g, self.follower.version), group_key=f"r{rid}-g{g}")

    def sleep(self, _dt):
        self.sleeps += 1
        self.on_sleep()

    def names(self):
        return [e for e, _ in self.events]


def test_sampler_loads_releases_and_waits_at_the_backlog_cap(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    sampler = _Sampler(tmp_path, releases, max_backlog_groups=4, rounds=5)
    trainer_reads = iter([(1, 4), (2, 8)])  # (version, groups read) per release

    def trainer_runs():  # the trainer reads the backlog and releases a new version
        version, read = next(trainer_reads)
        releases.release(version, f"h{version}", _write_weights, buffer_selected=read)

    sampler.on_sleep = trainer_runs
    assert sampler.loop.run() == 5
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2)
    assert [e.behavior_version for e in buffer._entries] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert sampler.served == [1, 2]
    assert sampler.names().count("rl_sampler_backlog_full") == 2
    assert buffer.data_ended and sampler.names()[-1] == "rl_sampler_data_end"


def test_sampler_keeps_the_last_version_while_the_trainer_is_stopped(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    releases.release(3, "h3", _write_weights, buffer_selected=0)
    sampler = _Sampler(tmp_path, releases, max_backlog_groups=4, rounds=5)
    assert sampler.loop.run() == 2  # the cap holds; stopped while waiting
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2)
    assert {e.behavior_version for e in buffer._entries} == {3} and len(buffer) == 4
    assert sampler.names()[-1] == "rl_sampler_stopped" and not buffer.data_ended


# ---------------------------------------------------------------- engine load
class _Engine:
    def __init__(self, log, name, version_after=None):
        self.log, self.server_url, self.version, self.version_after = log, name, None, version_after

    async def update_weights_from_disk(self, model_path, weight_version=None):
        self.log.append(("load", self.server_url, model_path, weight_version))

    async def update_weight_version(self, version):
        self.log.append(("stamp", self.server_url, version))
        self.version = self.version_after or version

    async def get_weight_version(self):
        return self.version


class _Controller:
    def __init__(self, log, engines):
        self.log, self.engines = log, engines

    async def start_update_weights(self):
        self.log.append(("lock",))
        return SimpleNamespace(rollout_engines=self.engines, snapshot_cell_id_to_hashes={"c0": "x"})

    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        self.log.append(("end", snapshot_cell_id_to_hashes))

    async def abort_update_weights(self):
        self.log.append(("abort",))


def _loader(log, engines):
    executor = SimpleNamespace(set_weight_version=lambda v: log.append(("executor", v)))
    return si.EngineReleaseLoader(_Controller(log, engines), executor, asyncio.run)


def test_engine_load_runs_under_the_update_lock_in_publisher_order(tmp_path):
    log = []
    loader = _loader(log, [_Engine(log, "e0"), _Engine(log, "e1")])
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    release = releases.release(2, "h2", _write_weights)
    loader(release, releases.resolve(release))
    path = str(releases.resolve(release))
    assert log == [("lock",), ("load", "e0", path, "2"), ("load", "e1", path, "2"),
                   ("stamp", "e0", "2"), ("stamp", "e1", "2"), ("end", {"c0": "x"}),
                   ("executor", 2)]
    log.clear()
    loader.stamp(0)  # base weights: version only
    assert [step[0] for step in log] == ["lock", "stamp", "stamp", "end", "executor"]


def test_engine_load_aborts_the_lock_when_an_engine_disagrees(tmp_path):
    log = []
    loader = _loader(log, [_Engine(log, "e0"), _Engine(log, "e1", version_after="9")])
    with pytest.raises(br.SampleBufferRolloutError, match="1/2 rollout engines"):
        loader.stamp(1)
    assert log[-1] == ("abort",) and ("executor", 1) not in log
    with pytest.raises(br.SampleBufferRolloutError, match="no rollout engines"):
        _loader([], []).stamp(1)


# ---------------------------------------------------------------- trainer releases
class _State:
    def __init__(self, version, digest):
        self.policy_version, self._digest = version, digest

    def policy_tensor_hash(self):
        return self._digest


def test_releasing_publisher_releases_every_publish(tmp_path):
    published, exported, events = [], [], []
    inner = SimpleNamespace(publish=lambda state: published.append(state.policy_version) or "result",
                            verify_serving_policy="inner-attr")
    positions = iter([None, 6, None])
    publisher = si.ReleasingPublisher(
        inner, PolicyReleaseStore(CheckpointStore(tmp_path)),
        export=lambda v, d: exported.append((v, d.name)) or _write_weights(d),
        read_position=lambda: next(positions), emit=lambda e, **f: events.append((e, f)))
    assert publisher.publish(_State(0, "h0")) == "result"
    publisher.publish(_State(1, "h1"))
    publisher.publish(_State(1, "h1"))  # republished after a restart: no new release
    latest = PolicyReleaseStore(CheckpointStore(tmp_path)).latest()
    assert published == [0, 1, 1] and exported == [(0, "r1-v0"), (1, "r2-v1")]
    assert (latest.version, latest.buffer_selected) == (1, 6)
    assert [f["seq"] for e, f in events if e == "rl_policy_released"] == [1, 2]
    assert publisher.verify_serving_policy == "inner-attr"


def test_wire_policy_releases_only_on_a_trainer_island(tmp_path):
    driver = SimpleNamespace(publisher=SimpleNamespace(publish=lambda s: None),
                             emit=lambda event, **fields: None,
                             rollout=SimpleNamespace(_in_executor=lambda fn, *a: fn(None, *a)))
    assert si.wire_policy_releases(driver, miles_args=SimpleNamespace(), actor_model=None,
                                   runner=None) is None
    args = SimpleNamespace(yeto_rl_sample_buffer=sample_roles.SampleBufferConfig(
        role="trainer", store=str(tmp_path)))
    calls = []
    actor = SimpleNamespace(export_hf=lambda rollout_id, path: calls.append((rollout_id, path)))
    runner = SimpleNamespace(run=lambda value: value)
    wired = si.wire_policy_releases(driver, miles_args=args, actor_model=actor, runner=runner)
    assert driver.publisher is wired
    br._ACTIVE.pop(br.TRAINER, None)
    wired.publish(_State(0, "h0"))
    assert calls == [(0, str(tmp_path / RELEASE_DIR / "r1-v0"))]


# ---------------------------------------------------------------- group identity
def test_groups_from_restarted_samplers_get_unique_buffer_identities(tmp_path):
    """Two sampler processes (a restart) both number their groups from 0; the
    trainer's batch has unique group and sample indexes, the same on a re-read."""
    for sampler_id in ("island0-a", "island0-b"):
        writer = br.GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2), sampler_id)
        for g in range(2):
            writer.write(_group(g, 1), group_key=f"r0-g{g}")
    reader = br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=4),
                            max_policy_age=1, wait_s=0.0, poll_s=0.0)
    taken, _ = reader.take(0, 4, current_version=1)
    groups = [br.renumber_group(g, position=pos, samples_per_group=2) for pos, g in taken]
    assert [s["group_index"] for g in groups for s in g] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [s["index"] for g in groups for s in g] == list(range(8))
    assert groups[2][0]["metadata"]["yeto_sampler_identity"] == {"group_index": 0, "index": 0}
    reader.reset(0)
    again, _ = reader.take(1, 4, current_version=1)
    assert [pos for pos, _ in again] == [pos for pos, _ in taken]
    with pytest.raises(br.SampleBufferRolloutError, match="samples per group 1"):
        br.renumber_group(taken[0][1], position=0, samples_per_group=1)
