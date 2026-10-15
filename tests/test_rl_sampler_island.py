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
    LORA,
    RELEASE_DIR,
    PolicyFollower,
    PolicyReleaseError,
    PolicyReleaseStore,
    directory_digest,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def _write_weights(directory: Path) -> None:
    (directory / "model.safetensors").write_text("w")


def _write_adapter(directory: Path, value: str = "a") -> None:
    (directory / "adapter_config.json").write_text('{"r": 8}')
    (directory / "adapter_model.safetensors").write_text(value)


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
    ({"rl_sampler_min_round_seconds": 0.0}, "a value > 0"),
    ({"rl_sample_role": "trainer", "rl_max_policy_age": 1, "rl_sampler_min_round_seconds": 5.0},
     "--rl-sampler-min-round-seconds needs --rl-sample-role sampler"),
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


def test_sampler_min_round_seconds_is_forwarded_and_resolved_for_the_sampler_only():
    args = _launch_args(rl_sampler_min_round_seconds=240.0)
    sample_roles.check_launch(args)
    assert " --rl-sampler-min-round-seconds 240.0" in sample_roles.island_flags(args)
    island = SimpleNamespace(rl_sample_role="sampler", rl_resume_store="/mnt/store",
                             rl_sampler_min_round_seconds=240.0)
    assert sample_roles.resolve(island, groups_per_round=8).min_round_s == 240.0
    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:trainer"],
                           rl_sampler_min_round_seconds=240.0, rl_sample_wait_s=9.0,
                           rl_sampler_max_backlog_rounds=None)
    assert sample_roles.island_args(pair, 1, 2).rl_sampler_min_round_seconds is None
    assert sample_roles.island_args(pair, 0, 2).rl_sampler_min_round_seconds == 240.0


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


def test_miles_argv_of_a_lora_sampler_and_trainer():
    from tests.test_rl_miles_adapter_config import flag_value, make_config, sub
    from yeto.rl.adapters.miles import config as mc

    base = mc.translate_run_config(make_config(), _tolerant())
    assert flag_value(base.argv, "--lora-rank") is not None

    def buffer_config(role, **kw):
        return dataclasses.replace(make_config(), sample_buffer=sample_roles.SampleBufferConfig(
            role=role, store="/mnt/store", **kw))

    sampler = mc.translate_run_config(buffer_config("sampler", max_backlog_groups=4), _tolerant())
    assert [a for a in sampler.argv if a not in base.argv] == [
        "--rollout-function-path", br.ROLLOUT_FUNCTION_PATHS["sampler"], "--debug-rollout-only"]
    assert flag_value(sampler.argv, "--lora-rank") == flag_value(base.argv, "--lora-rank")
    two_gpus = buffer_config("trainer")
    assert two_gpus.parallel.colocated and mc.placement_request(two_gpus).trainer_gpus == 2
    one_gpu = sub(two_gpus, "parallel", actor_num_gpus_per_node=1)
    colocated = mc.translate_run_config(one_gpu, _tolerant(), max_policy_age=1)
    assert flag_value(colocated.argv, "--save-hf") == mc.RELEASE_EXPORTER_SAVE_HF
    with pytest.raises(mc.MilesConfigError, match="one trainer GPU only"):
        mc.translate_run_config(two_gpus, _tolerant(), max_policy_age=1)
    partitioned = dataclasses.replace(sub(buffer_config("trainer"), "serving", offload_train=False),
                                      parallel=make_config(colocated=False).parallel)
    trainer = mc.translate_run_config(partitioned, _tolerant(), max_policy_age=1)
    assert flag_value(trainer.argv, "--save-hf") == mc.RELEASE_EXPORTER_SAVE_HF


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


def test_lora_release_records_the_file_digest_and_the_follower_checks_it(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    with pytest.raises(ValueError, match="weights must be one of"):
        releases.release(1, "h1", _write_adapter, weights="delta")
    full = releases.release(1, "h1", _write_weights)
    assert (full.weights, full.sha256, full.engine_checksum) == ("full", None, None)
    lora = releases.release(2, "h2", _write_adapter, weights=LORA, engine_checksum="c2")
    directory = releases.resolve(lora)
    assert (lora.weights, lora.engine_checksum) == (LORA, "c2")
    assert lora.sha256 == directory_digest(directory) and releases.latest() == lora
    other = tmp_path / "other"
    other.mkdir()
    _write_adapter(other, "b")
    assert directory_digest(other) != lora.sha256  # the bytes count, not only the names
    loaded = []
    follower = PolicyFollower(releases, lambda r, d: loaded.append(r.seq))
    (directory / "adapter_model.safetensors").write_text("tampered")
    with pytest.raises(PolicyReleaseError, match="weight files sha256"):
        follower.refresh()
    assert loaded == [] and follower.release is None
    _write_adapter(directory)
    assert follower.refresh().seq == 2 and loaded == [2]


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


def test_lora_sampler_waits_for_the_first_release(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    sampler = _Sampler(tmp_path, releases, max_backlog_groups=100, rounds=2)
    sampler.loop.require_release = True
    sampler.loop.loaded_fields = lambda: {"served_checksum": "s"}
    sampler.on_sleep = lambda: releases.release(0, "h0", _write_adapter, weights=LORA)
    assert sampler.loop.run() == 2
    assert sampler.sleeps == 1 and sampler.served == [0]
    names = sampler.names()
    assert names[:3] == ["rl_sampler_waiting_policy", "rl_sampler_policy_loaded",
                         "rl_sampler_policy_ready"]
    loaded = dict(sampler.events)["rl_sampler_policy_loaded"]
    assert (loaded["weights"], loaded["served_checksum"], loaded["seq"]) == (LORA, "s", 1)
    assert loaded["sha256"] == releases.latest().sha256


def test_sampler_throttle_keeps_min_round_seconds_between_round_starts(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    sampler = _Sampler(tmp_path, releases, max_backlog_groups=100, rounds=3)
    now = [0.0]
    slept = []
    sampler.loop.clock = lambda: now[0]
    sampler.loop.min_round_s = 10.0
    sampler.loop.sleep = slept.append

    def generate(rid, inner=sampler.generate):
        now[0] += 4.0  # a round takes 4 s
        inner(rid)

    sampler.loop.generate = generate
    assert sampler.loop.run() == 3
    assert slept == [6.0, 6.0]  # not after the last round
    assert sampler.names().count("rl_sampler_throttle") == 2


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


def _lora_loader(log, engines, bodies, staging=None):
    async def post(engine, endpoint, payload):
        log.append((endpoint, engine.server_url, payload.get("lora_path")))
        return {"success": True}

    async def warmup(engine, adapter, timeout_s):
        log.append(("warmup", engine.server_url, adapter))

    controller = _Controller(log, engines)

    async def check_weights(action):
        log.append(("checksum",))
        return bodies

    controller.check_weights = check_weights
    executor = SimpleNamespace(set_weight_version=lambda v: log.append(("executor", v)))
    async def probe(engine, adapter):
        log.append(("probe", engine.server_url))
        return {"logprob_sum": -1.5}

    return si.EngineReleaseLoader(controller, executor, asyncio.run, lora=True, post=post,
                                  warmup=warmup, flatten=lambda raw: raw, probe=probe,
                                  staging_root=staging)


def test_lora_engine_load_hot_swaps_the_adapter_and_reads_it_back(tmp_path):
    from yeto.rl.adapters.miles.publish import weight_view_digest

    log = []
    body = {"model.w": "base", "lora:miles_lora:q_proj:0:A": "a1"}
    staging = tmp_path / "staging"
    staging.mkdir()
    loader = _lora_loader(log, [_Engine(log, "e0"), _Engine(log, "e1")], [body, dict(body)],
                          staging=str(staging))
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    expected = weight_view_digest([body], lora=True)
    assert expected == weight_view_digest([{"lora:miles_lora:q_proj:0:A": "a1"}], lora=False)
    first = releases.release(0, "h0", _write_adapter, weights=LORA, engine_checksum=expected)
    loader(first, releases.resolve(first))
    # the engines load a local copy: no file on the store stays open (Modal Volume reload)
    (path,) = [str(p) for p in staging.iterdir()]
    assert (Path(path) / "adapter_model.safetensors").read_text() == "a"
    assert log == [("lock",), ("load_lora_adapter", "e0", path), ("load_lora_adapter", "e1", path),
                   ("stamp", "e0", "0"), ("stamp", "e1", "0"), ("end", {"c0": "x"}),
                   ("executor", 0), ("warmup", "e0", "miles_lora"), ("warmup", "e1", "miles_lora"),
                   ("checksum",), ("probe", "e0")]
    assert loader.last_checksum == expected and loader.last_probe == {"logprob_sum": -1.5}
    log.clear()
    second = releases.release(1, "h1", _write_adapter, weights=LORA)  # no trainer checksum
    loader(second, releases.resolve(second))
    assert [step[0] for step in log[:5]] == ["lock", "unload_lora_adapter", "unload_lora_adapter",
                                            "load_lora_adapter", "load_lora_adapter"]
    (served,) = [str(p) for p in staging.iterdir()]  # the v0 copy was removed after the unload
    assert served != path and log[3][2] == served
    third = releases.release(2, "h2", _write_adapter, weights=LORA, engine_checksum="other")
    with pytest.raises(br.SampleBufferRolloutError, match="the trainer published other"):
        loader(third, releases.resolve(third))
    full = releases.release(3, "h3", _write_weights)
    with pytest.raises(br.SampleBufferRolloutError, match="holds full weights"):
        loader(full, releases.resolve(full))


def test_probe_generate_digests_the_greedy_answer(monkeypatch):
    sent = []

    async def post(engine, endpoint, payload, timeout_s):
        sent.append((endpoint, payload["lora_path"], payload["sampling_params"]["temperature"]))
        return {"text": " 72", "meta_info": {"output_token_logprobs": [[-0.5, 1, None],
                                                                       [-0.25, 2, None]]}}

    monkeypatch.setattr(si, "_post_engine", post)
    answer = asyncio.run(si.probe_generate(SimpleNamespace(server_url="e0"), "miles_lora"))
    assert sent == [("generate", "miles_lora", 0.0)]
    assert (answer["logprob_sum"], answer["tokens"], answer["text"]) == (-0.75, 2, " 72")


def test_engines_without_a_served_adapter_give_no_lora_checksum():
    from yeto.rl.adapters.miles.publish import weight_view_digest

    assert weight_view_digest([{"model.w": "base"}], lora=True) is None
    assert weight_view_digest([], lora=False) is None
    lora = {"lora:miles_lora:q_proj:0:A": "a1"}
    assert weight_view_digest([lora, {"lora:miles_lora:q_proj:0:A": "a2"}], lora=True) is None
    assert weight_view_digest([{"model.w": "x"}], lora=False) is not None


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


def test_releasing_publisher_records_the_lora_adapter_checksum(tmp_path):
    from yeto.rl.adapters.miles.publish import weight_view_digest

    body = {"model.w": "base", "lora:miles_lora:q_proj:0:A": "a1"}
    inner = SimpleNamespace(publish=lambda state: None,
                            last_engine_checksums={"e0": body, "e1": dict(body)})
    events = []
    publisher = si.ReleasingPublisher(
        inner, PolicyReleaseStore(CheckpointStore(tmp_path)),
        export=lambda v, d: _write_adapter(d), read_position=lambda: 0,
        emit=lambda e, **f: events.append((e, f)), weights=LORA)
    publisher.publish(_State(0, "h0"))
    latest = PolicyReleaseStore(CheckpointStore(tmp_path)).latest()
    assert latest.weights == LORA and latest.engine_checksum == weight_view_digest([body], lora=True)
    assert events[0][1]["sha256"] == latest.sha256
    inner.last_engine_checksums = {"e0": {"model.w": "base"}}  # adapter never served
    publisher.publish(_State(1, "h1"))
    assert PolicyReleaseStore(CheckpointStore(tmp_path)).latest().engine_checksum is None


def test_wire_policy_releases_exports_only_the_adapter_in_lora_mode(tmp_path):
    from yeto.rl.adapters.miles.release_plugin import WRITE_LORA_ADAPTER

    driver = SimpleNamespace(publisher=SimpleNamespace(publish=lambda s: None),
                             emit=lambda event, **fields: None,
                             rollout=SimpleNamespace(_in_executor=lambda fn, *a: fn(None, *a)))
    args = SimpleNamespace(lora_rank=8, yeto_rl_sample_buffer=sample_roles.SampleBufferConfig(
        role="trainer", store=str(tmp_path)))
    calls = []

    def run_plugin(fn_path, kwargs):
        calls.append((fn_path, kwargs))
        _write_adapter(Path(kwargs["path"]))

    actor = SimpleNamespace(run_plugin=run_plugin)
    wired = si.wire_policy_releases(driver, miles_args=args, actor_model=actor,
                                    runner=SimpleNamespace(run=lambda value: value))
    br._ACTIVE.pop(br.TRAINER, None)
    wired.publish(_State(0, "h0"))
    assert calls == [(WRITE_LORA_ADAPTER, {"path": str(tmp_path / RELEASE_DIR / "r1-v0")})]
    assert PolicyReleaseStore(CheckpointStore(tmp_path)).latest().weights == LORA


def test_release_plugin_writes_the_adapter_with_the_snapshot_publisher(tmp_path):
    from yeto.rl.adapters.miles import release_plugin

    written = []
    actor = SimpleNamespace(snapshot_publisher=SimpleNamespace(
        write_adapter=lambda adapter, path: written.append((adapter, path))))
    assert release_plugin.write_lora_adapter(actor, path=str(tmp_path)) == {"path": str(tmp_path)}
    assert written == [(None, tmp_path)]
    # an asleep colocated actor: groups reloaded, memory saver disabled, groups destroyed
    import sys
    calls = []

    class _Saver:
        def disable(self):
            import contextlib

            @contextlib.contextmanager
            def cm():
                calls.append("disable")
                yield
                calls.append("enable")
            return cm()

    fake_pg = SimpleNamespace(reload_process_groups=lambda: calls.append("reload"),
                              destroy_process_groups=lambda: calls.append("destroy"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "miles", SimpleNamespace())
        mp.setitem(sys.modules, "miles.utils", SimpleNamespace())
        mp.setitem(sys.modules, "miles.utils.reloadable_process_group", fake_pg)
        mp.setitem(sys.modules, "torch_memory_saver", SimpleNamespace(torch_memory_saver=_Saver()))
        asleep = SimpleNamespace(args=SimpleNamespace(offload_train=True), _asleep=True,
                                 snapshot_publisher=SimpleNamespace(
                                     write_adapter=lambda a, p: calls.append("write")))
        release_plugin.write_lora_adapter(asleep, path=str(tmp_path))
    assert calls == ["reload", "disable", "write", "enable", "destroy"]
    with pytest.raises(RuntimeError, match="snapshot publisher"):
        release_plugin.write_lora_adapter(SimpleNamespace(snapshot_publisher=None), path="x")


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


def test_sampler_debug_rollout_only_is_not_a_placement_rewrite():
    """L5.4 GPU run s24-l54-pair-20261014e: a fixed-partition sampler island asked
    for --debug-rollout-only and was refused as a rewritten placement."""
    import dataclasses

    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.adapters.miles.placement import PlacementRewriteError

    launch = mc.translate_run_config(_make_miles_config(), _algorithm_spec())
    ns = SimpleNamespace(use_session_server=False, partial_rollout=False, indep_dp=False,
                         debug_rollout_only=True, colocate=launch.placement.kind == "colocated",
                         actor_num_nodes=1, actor_num_gpus_per_node=launch.placement.trainer_gpus,
                         rollout_num_gpus=launch.placement.rollout_gpus)
    with pytest.raises(PlacementRewriteError, match="debug_rollout_only=True"):
        mc.validate_parsed_args(ns, launch, num_cells=lambda _a: 1)
    sampler = dataclasses.replace(launch, argv=(*launch.argv, mc.SAMPLER_ONLY_FLAG))
    mc.validate_parsed_args(ns, sampler, num_cells=lambda _a: 1)


def _make_miles_config():
    from tests.test_rl_miles_adapter_config import make_config

    return make_config()


def _algorithm_spec():
    from yeto.rl.engine.algorithm import AlgorithmSpec

    return AlgorithmSpec()


def test_ray_metadata_sink_discard_drops_every_record():
    from yeto.rl.adapters.miles.rollout import RayMetadataSink

    class _Actor:
        class take_all:
            @staticmethod
            def remote():
                return ["a", "b"]

    sink = RayMetadataSink.__new__(RayMetadataSink)
    sink._ray = SimpleNamespace(get=lambda x: x)
    sink._actor = _Actor()
    assert sink.discard() == 2
