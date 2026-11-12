"""A weight load that failed after it touched the engine taints the trials that
ran across it (CPU).

A forced load (``reload_max_defer_s``) runs while a streaming agentic round still
holds trials on the engine. When that in-place load fails part way (an engine
error, OOM, a timeout, a failed post-load check), the engine keeps reporting the
old ``served_version`` but holds weights that are neither the old nor the new
release. Trials that generated across the failure must not be written under the
old version: their tokens did not come from it (off-policy correction would use
the wrong behavior policy).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.engine.agentic_round import GROUP_EVENT
from yeto.rl.engine.policy_release import PolicyRelease, ReleaseNotLoaded
from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks

from test_agentic_round_streaming import Harness, Prompts, _entries, _round, _wait


def _loader(engine, events):
    checks = PostLoadChecks(expected_expert_matrices=None, require_fixed_answers=False)
    loader = PortReleaseLoader(engine, checks, lambda n, **f: events.append((n, f)), max_consecutive_failures=5)
    return loader


def _release(tmp_path, version):
    directory = tmp_path / f"r{version}"
    directory.mkdir(exist_ok=True)
    return PolicyRelease(seq=version, version=version, policy_hash=f"h{version}", path=f"r{version}"), directory


def test_a_group_that_ran_across_a_failed_in_place_load_is_not_written(tmp_path):
    harness = Harness({"b": 0.0})
    harness.gates["a"] = threading.Event()
    engine = FakeEngine()
    engine.start(None, 0)
    events_l: list = []
    loader = _loader(engine, events_l)
    loader.start(None, 0)
    rnd, engine, events = _round(tmp_path / "buf", harness, Prompts({0: ["a"]}), engine=engine)
    rnd.weights_state = loader.weights_state
    try:
        rnd(0)
        _wait(lambda: len(harness.started) == 2)
        # the forced load (reload_max_defer_s) while group a still runs: it fails in the engine
        engine.fail_load = "CUDA out of memory while copying layer 31"
        release, directory = _release(tmp_path, 1)
        with pytest.raises(ReleaseNotLoaded):
            loader(release, directory)
        assert engine.served_version() == 0 and not loader.ready  # label unchanged, weights unknown
        harness.gates["a"].set()
        rnd.flush()
    finally:
        rnd.close()
    assert not _entries(tmp_path / "buf")[0]  # nothing written as v0 from the unchecked weights
    (group,) = [f for n, f in events if n == GROUP_EVENT]
    assert not group["written"] and group["dropped_reason"] == "unchecked_weights_in_flight"


def test_a_refused_release_does_not_taint_the_running_groups(tmp_path):
    harness = Harness()
    harness.gates["a"] = threading.Event()
    engine = FakeEngine()
    engine.start(None, 0)
    loader = _loader(engine, [])
    loader.start(None, 0)
    loader.expected_selection_sha256 = "aa"  # the release names no selection: refused before the engine
    rnd, engine, events = _round(tmp_path / "buf", harness, Prompts({0: ["a"]}), engine=engine)
    rnd.weights_state = loader.weights_state
    try:
        rnd(0)
        _wait(lambda: len(harness.started) == 2)
        release, directory = _release(tmp_path, 1)
        with pytest.raises(ReleaseNotLoaded):
            loader(release, directory)
        assert loader.ready
        harness.gates["a"].set()
        rnd.flush()
    finally:
        rnd.close()
    assert [e.behavior_version for e in _entries(tmp_path / "buf")[0]] == [0]


def test_a_group_started_after_a_successful_restart_is_written(tmp_path):
    harness = Harness({"b": 0.0})
    engine = FakeEngine()
    engine.start(None, 0)
    loader = _loader(engine, [])
    loader.start(None, 0)
    rnd, engine, events = _round(tmp_path / "buf", harness, Prompts({0: [], 1: ["b"]}), engine=engine)
    rnd.weights_state = loader.weights_state
    try:
        engine.fail_load = "boom"
        release, directory = _release(tmp_path, 1)
        with pytest.raises(ReleaseNotLoaded):
            loader(release, directory)
        loader(release, directory)  # the retry restarts the engine and passes its checks
        assert loader.ready and engine.served_version() == 1
        rnd(1)
        rnd.flush()
    finally:
        rnd.close()
    assert [e.behavior_version for e in _entries(tmp_path / "buf")[0]] == [1]


def test_a_group_started_while_the_engine_holds_unchecked_weights_is_not_written(tmp_path):
    harness = Harness({"b": 0.0})
    engine = FakeEngine()
    engine.start(None, 0)
    loader = _loader(engine, [])
    loader.start(None, 0)
    rnd, engine, events = _round(tmp_path / "buf", harness, Prompts({0: ["b"]}), engine=engine)
    rnd.weights_state = loader.weights_state
    try:
        engine.fail_load = "boom"
        release, directory = _release(tmp_path, 1)
        with pytest.raises(ReleaseNotLoaded):
            loader(release, directory)
        rnd(0)  # the trials start before the engine is restarted
        rnd.flush()
    finally:
        rnd.close()
    assert not _entries(tmp_path / "buf")[0]


def test_build_engine_sampler_wires_the_loader_weights_state_into_a_streaming_round(tmp_path):
    from test_agentic_round_streaming import Recorder
    from yeto.rl.engine.agentic_round import AgenticRound
    from yeto.rl.engine.engine_sampler import EngineSamplerConfig, build_engine_sampler
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.engine.sampler_engine import SamplingParams

    store = CheckpointStore(tmp_path)
    config = EngineSamplerConfig(sampler_id="s", num_rollouts=1, max_backlog_groups=1,
                                 sampling=SamplingParams(max_tokens=8, n=2), max_prompt_tokens=16, poll_s=0.0)

    def make(engine_, writer, **kw):
        return AgenticRound(engine_, writer, recorder=Recorder(engine_), harness=Harness(), max_in_flight=2,
                            max_response_len=64, min_samples=1, task_defaults={"provider": "docker"}, **kw)

    island = build_engine_sampler(config, engine=FakeEngine(), releases=PolicyReleaseStore(store),
                                  buffer=FileSampleBuffer(store, batch_size=1), emit=lambda n, **f: None,
                                  prompts=lambda r: [], reward=lambda i, c: 0.0, make_round=make,
                                  checks=PostLoadChecks(expected_expert_matrices=None, require_fixed_answers=False))
    try:
        assert island.round.weights_state == island.loader.weights_state
        assert island.round.weights_state() == (True, 0)
    finally:
        island.round.close()
