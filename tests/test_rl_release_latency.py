"""S26 E5 (CPU, no Ray, no Miles): the phase timings of a policy release (trainer
side and sampler side) and the fused LoRA stage (``--rl-sampler-fused-stage``)."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_sampler_island import (
    _Engine,
    _launch_args,
    _lora_loader,
    _State,
    _write_adapter,
    _write_weights,
)
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.engine import sample_roles
from yeto.rl.engine.policy_release import (
    LORA,
    RECORD,
    PhaseTimer,
    PolicyFollower,
    PolicyReleaseError,
    PolicyReleaseStore,
    copy_with_digest,
    directory_digest,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop


def _ticks(step: float = 0.5):
    """A clock that advances ``step`` seconds per read."""
    counter = itertools.count()
    return lambda: next(counter) * step


def test_phase_timer_adds_each_phase_in_first_use_order():
    timer = PhaseTimer(_ticks())
    with timer.phase("b"):
        pass
    with timer.phase("a"):
        pass
    with timer.phase("b"):
        pass
    timer.add("c", 0.25)
    assert timer.as_dict() == {"b": 1.0, "a": 0.5, "c": 0.25}
    with pytest.raises(RuntimeError), timer.phase("d"):
        raise RuntimeError("boom")
    assert timer.as_dict()["d"] == 0.5  # a failed phase still counts


def test_copy_with_digest_copies_everything_and_equals_the_directory_digest(tmp_path):
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"\x00\x01" * 70_000)  # > one chunk
    (source / "sub" / "extra.json").write_text("{}")
    (source / RECORD).write_text('{"seq": 1}')
    target = tmp_path / "dst"
    digest = copy_with_digest(source, target)
    assert digest == directory_digest(source) == directory_digest(target)
    copied = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert copied == [RECORD, "adapter_model.safetensors", "sub/extra.json"]
    assert (target / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors").read_bytes()
    (source / RECORD).write_text('{"seq": 2}')  # the record is not digested
    assert copy_with_digest(source, tmp_path / "dst2") == digest


def test_release_records_the_trainer_phases(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path), clock=lambda: 7.0, timer=_ticks(),
                                  cached_latest=False)
    assert releases.last_timings == {}
    lora = releases.release(1, "h1", _write_adapter, weights=LORA)
    assert lora.released_at == 7.0
    assert list(releases.last_timings) == ["read_latest_s", "export_s", "digest_s",
                                           "commit_weights_s", "commit_latest_s", "prune_s"]
    assert all(value == 0.5 for value in releases.last_timings.values())
    before = dict(releases.last_timings)
    assert releases.release(1, "h1", _write_adapter, weights=LORA) is None  # republish
    assert releases.last_timings == before  # only a written release sets the timings


def test_follower_records_its_phases_and_when_it_first_saw_a_release(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    wall = iter([100.0, 200.0, 300.0])
    loaded = []
    follower = PolicyFollower(releases, lambda r, d: loaded.append(r.seq), timer=_ticks(),
                              wall=lambda: next(wall))
    assert follower.refresh() is None and follower.seen_at is None
    releases.release(1, "h1", _write_adapter, weights=LORA)
    assert follower.refresh().seq == 1 and follower.seen_at == 100.0
    assert list(follower.last_timings) == ["read_latest_s", "resolve_s", "verify_s", "load_s"]
    assert follower.refresh() is None and follower.seen_at == 100.0  # same release: same time
    releases.release(2, "h2", _write_adapter, weights=LORA)
    assert follower.refresh().seq == 2 and follower.seen_at == 200.0 and loaded == [1, 2]


def test_follower_leaves_the_file_check_to_a_hook_that_verifies_files(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    release = releases.release(1, "h1", _write_adapter, weights=LORA)
    weights = releases.resolve(release) / "adapter_model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)  # same size, other bytes
    with pytest.raises(PolicyReleaseError, match="weight files sha256"):
        PolicyFollower(releases, lambda r, d: None).refresh()

    class Hook:
        verifies_files = True

        def __init__(self):
            self.calls = []

        def __call__(self, release, directory):
            self.calls.append(release.seq)

    hook = Hook()
    follower = PolicyFollower(releases, hook)
    assert follower.refresh().seq == 1 and hook.calls == [1]
    assert "verify_s" not in follower.last_timings


def _fused(log, staging, fused=True):
    body = {"model.w": "base", "lora:miles_lora:q_proj:0:A": "a1"}
    loader = _lora_loader(log, [_Engine(log, "e0"), _Engine(log, "e1")], [body, dict(body)],
                          staging=str(staging))
    loader.fused_stage = fused
    return loader


def test_fused_stage_loads_like_the_plain_copy_and_records_the_load_phases(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    plain_log, fused_log = [], []
    plain, fused = _fused(plain_log, staging, fused=False), _fused(fused_log, staging)
    assert (plain.verifies_files, fused.verifies_files) == (False, True)
    release = releases.release(0, "h0", _write_adapter, weights=LORA)
    PolicyFollower(releases, fused).refresh()
    (path,) = [str(p) for p in staging.iterdir()]
    assert (Path(path) / "adapter_model.safetensors").read_text() == "a"
    plain(release, releases.resolve(release))
    plain_path = str(plain._staged)
    # the same engine calls in the same order; only the local copy's name differs
    assert [step[:2] for step in fused_log] == [step[:2] for step in plain_log]
    assert ("load_lora_adapter", "e0", path) in fused_log
    assert ("load_lora_adapter", "e0", plain_path) in plain_log
    assert set(fused.last_timings) == {"stage_s", "lock_s", "engine_load_s", "stamp_s",
                                       "unlock_s", "warmup_s", "checksum_s", "probe_s"}
    assert fused.last_checksum == plain.last_checksum is not None


def test_fused_stage_refuses_changed_bytes_before_any_engine_call(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    release = releases.release(0, "h0", _write_adapter, weights=LORA)
    weights = releases.resolve(release) / "adapter_model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    log = []
    loader = _fused(log, staging)
    follower = PolicyFollower(releases, loader)
    with pytest.raises(PolicyReleaseError, match="weight files sha256"):
        follower.refresh()
    assert log == [] and list(staging.iterdir()) == []  # no lock, no engine, no copy left
    assert follower.release is None and "stage_s" in loader.last_timings


def test_full_parameter_loader_ignores_the_fused_stage(tmp_path):
    loader = si.EngineReleaseLoader(None, None, None, fused_stage=True)
    assert (loader.fused_stage, loader.verifies_files) == (False, False)


def test_a_failed_engine_load_removes_the_unused_copy(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    release = releases.release(0, "h0", _write_adapter, weights=LORA)
    log = []
    loader = _fused(log, staging)

    async def failing_post(engine, endpoint, payload):
        raise RuntimeError("engine down")

    loader.post = failing_post
    with pytest.raises(RuntimeError, match="engine down"):
        loader(release, releases.resolve(release))
    assert list(staging.iterdir()) == [] and log[-1] == ("abort",)


def test_sampler_loaded_event_carries_the_latency_fields(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path), clock=lambda: 50.0)
    follower = PolicyFollower(releases, lambda r, d: None, wall=lambda: 60.0)
    events = []
    loop = SamplerLoop(follower=follower, buffer=FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2),
                       generate=lambda r: None, num_rollouts=1, max_backlog_groups=4,
                       emit=lambda e, **f: events.append((e, f)),
                       loaded_fields=lambda: {"load_timings": {"stage_s": 0.1}})
    releases.release(1, "h1", _write_weights)
    loop.run()
    (loaded,) = [f for e, f in events if e == "rl_sampler_policy_loaded"]
    assert (loaded["released_at"], loaded["seen_at"]) == (50.0, 60.0)
    assert set(loaded["follower_timings"]) == {"read_latest_s", "resolve_s", "verify_s", "load_s"}
    assert loaded["load_timings"] == {"stage_s": 0.1}


def test_releasing_publisher_reports_the_publish_and_release_phases(tmp_path):
    events = []
    publisher = si.ReleasingPublisher(
        SimpleNamespace(publish=lambda state: None), PolicyReleaseStore(CheckpointStore(tmp_path)),
        export=lambda v, d: _write_adapter(d), read_position=lambda: 0,
        emit=lambda e, **f: events.append((e, f)), weights=LORA)
    publisher.publish(_State(0, "h0"))
    (released,) = [f for e, f in events if e == "rl_policy_released"]
    assert released["publish_s"] >= 0 and released["released_at"] > 0
    assert "export_s" in released["release_timings"]
    assert released["release_bytes"] == len('{"r": 8}') + len("a")


# ---------------------------------------------------------------- the launch flag
def test_fused_stage_flag_is_sampler_and_lora_only():
    sample_roles.check_launch(_launch_args(rl_sampler_fused_stage=True))
    for kw, message in (
            ({"rl_sample_role": "trainer", "rl_max_policy_age": 1}, "needs --rl-sample-role sampler"),
            ({"tuning": "full"}, "--tuning lora")):
        with pytest.raises(ValueError, match=message):
            sample_roles.check_launch(_launch_args(rl_sampler_fused_stage=True, **kw))
    with pytest.raises(ValueError, match="needs --rl-sample-role sampler"):
        sample_roles.check_launch(SimpleNamespace(rl_sampler_fused_stage=True))


def test_fused_stage_flag_is_forwarded_to_sampler_islands_only():
    args = _launch_args(rl_sampler_fused_stage=True)
    assert sample_roles.island_flags(args).endswith(" --rl-sampler-fused-stage")
    assert "--rl-sampler-fused-stage" not in sample_roles.island_flags(_launch_args())
    island = SimpleNamespace(rl_sample_role="sampler", rl_resume_store="/mnt/store",
                             rl_sampler_fused_stage=True)
    assert sample_roles.resolve(island, groups_per_round=8).fused_stage is True
    assert sample_roles.resolve(SimpleNamespace(rl_sample_role="sampler", rl_resume_store="/s"),
                                groups_per_round=8).fused_stage is False
    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:trainer"],
                           rl_sampler_fused_stage=True, rl_sampler_min_round_seconds=None,
                           rl_sample_wait_s=None, rl_sampler_max_backlog_rounds=None)
    assert sample_roles.island_args(pair, 0, 2).rl_sampler_fused_stage is True
    assert sample_roles.island_args(pair, 1, 2).rl_sampler_fused_stage is False


def test_fused_stage_flag_exists_on_both_layers():
    from yeto.rl.engine.run_flags import CLI, ISLAND, flags_for

    for layer in (CLI, ISLAND):
        parser = argparse.ArgumentParser()
        for flag in flags_for(layer):
            if flag.options[0] == "--rl-sampler-fused-stage":
                parser.add_argument(*flag.options, **dict(flag.kwargs))
        assert parser.parse_args([]).rl_sampler_fused_stage is False
        assert parser.parse_args(["--rl-sampler-fused-stage"]).rl_sampler_fused_stage is True


# ---------------------------------------------------------------- the writer's known LATEST
class _CountingStore(CheckpointStore):
    def __init__(self, root):
        super().__init__(root)
        self.reloads = self.commits = 0

    def reload(self):
        self.reloads += 1

    def commit(self):
        self.commits += 1
        return 0.0


def test_one_writer_reloads_only_for_its_first_release_and_commits_once(tmp_path):
    store = _CountingStore(tmp_path)
    releases = PolicyReleaseStore(store)
    first = releases.release(1, "h1", _write_adapter, weights=LORA)
    assert (releases.last_latest_source, store.reloads, store.commits) == ("reloaded", 1, 1)
    second = releases.release(2, "h2", _write_adapter, weights=LORA)
    assert (releases.last_latest_source, store.reloads, store.commits) == ("cached", 1, 2)
    assert "commit_s" in releases.last_timings and "commit_weights_s" not in releases.last_timings
    assert releases.release(2, "h2", _write_adapter, weights=LORA) is None  # republish: still known
    assert store.reloads == 1
    assert (first.seq, second.seq) == (1, 2) and PolicyReleaseStore(CheckpointStore(tmp_path)).latest() == second


def test_a_resumed_writer_reads_latest_again_and_continues_the_seq(tmp_path):
    PolicyReleaseStore(CheckpointStore(tmp_path)).release(1, "h1", _write_weights)
    store = _CountingStore(tmp_path)
    resumed = PolicyReleaseStore(store)  # a new process over the same store
    assert resumed.release(1, "h1", _write_weights) is None  # same version and hash: nothing
    assert (resumed.last_latest_source, store.reloads) == ("reloaded", 1)
    nxt = resumed.release(2, "h2", _write_weights)
    assert nxt.seq == 2 and resumed.last_latest_source == "cached"


def test_another_writer_is_detected_and_latest_is_read_again(tmp_path):
    store = _CountingStore(tmp_path)
    mine = PolicyReleaseStore(store)
    mine.release(1, "h1", _write_weights)
    PolicyReleaseStore(CheckpointStore(tmp_path)).release(2, "other", _write_weights)  # seq 2
    third = mine.release(3, "h3", _write_weights)
    assert (mine.last_latest_source, store.reloads) == ("other_writer", 2)
    assert third.seq == 3  # never reuses the other writer's seq


def test_a_failed_release_makes_the_next_one_read_latest_again(tmp_path):
    store = _CountingStore(tmp_path)
    releases = PolicyReleaseStore(store)
    releases.release(1, "h1", _write_weights)
    with pytest.raises(RuntimeError):
        releases.release(2, "h2", lambda d: (_ for _ in ()).throw(RuntimeError("export failed")))
    releases.release(2, "h2", _write_weights)
    assert releases.last_latest_source == "reloaded" and store.reloads == 2


def test_cached_latest_off_is_the_earlier_behavior(tmp_path):
    store = _CountingStore(tmp_path)
    releases = PolicyReleaseStore(store, cached_latest=False)
    releases.release(1, "h1", _write_weights)
    releases.release(2, "h2", _write_weights)
    assert (releases.last_latest_source, store.reloads, store.commits) == ("reloaded", 2, 4)


def test_reload_latest_flag_is_trainer_only_and_reaches_the_release_store(tmp_path):
    trainer = _launch_args(rl_sample_role="trainer", rl_max_policy_age=1, rl_release_reload_latest=True)
    sample_roles.check_launch(trainer)
    assert sample_roles.island_flags(trainer).endswith(" --rl-release-reload-latest")
    with pytest.raises(ValueError, match="only a trainer writes"):
        sample_roles.check_launch(_launch_args(rl_release_reload_latest=True))
    island = SimpleNamespace(rl_sample_role="trainer", rl_resume_store=str(tmp_path),
                             rl_release_reload_latest=True)
    config = sample_roles.resolve(island, groups_per_round=8)
    assert config.release_legacy is True
    assert si.release_store(config).cached_latest is False
    default = sample_roles.resolve(SimpleNamespace(rl_sample_role="trainer", rl_resume_store=str(tmp_path)),
                                   groups_per_round=8)
    assert si.release_store(default).cached_latest is True
    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:trainer"], rl_release_reload_latest=True,
                           rl_sampler_min_round_seconds=None, rl_sample_wait_s=None,
                           rl_sampler_max_backlog_rounds=None)
    assert sample_roles.island_args(pair, 0, 2).rl_release_reload_latest is False
    assert sample_roles.island_args(pair, 1, 2).rl_release_reload_latest is True
