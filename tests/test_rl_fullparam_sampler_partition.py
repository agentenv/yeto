"""Full-parameter sampler islands: their own engine partition, and the sampler
side of full-weight updates (release order, versions, staleness counters, the
in-flight rule). CPU only: fake engines, a local store, no Ray."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_s25_g1_stage1mini import _MINI_TEXT_CONFIG, REPO, glm5_next

from yeto.rl.engine.engine_sampler import _completion_version
from yeto.rl.engine.policy_release import FULL, PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.run_config import full_parameter_layout_problem
from yeto.rl.engine.sampler_engine import Completion, SamplerEngineError, updates_during_generation
from yeto.rl.engine.sampler_loop import SamplerLoop


# ------------------------------------------------------------------ partition rule
def test_full_parameter_sampler_island_is_its_own_engine_partition():
    # Any DP and node count: the sampler builds no trainer.
    assert full_parameter_layout_problem(data_parallel=4, actor_num_nodes=2, rollout_gpus=None,
                                         rollout_gpus_per_engine=2, sample_role="sampler",
                                         island_gpus=16) is None
    assert full_parameter_layout_problem(data_parallel=1, actor_num_nodes=1, rollout_gpus=None,
                                         rollout_gpus_per_engine=None, sample_role="sampler",
                                         island_gpus=1) is None


def test_full_parameter_sampler_refuses_a_separate_rollout_count_and_uneven_engines():
    assert "drop --rollout-num-gpus" in full_parameter_layout_problem(
        data_parallel=1, actor_num_nodes=1, rollout_gpus=1, rollout_gpus_per_engine=1,
        sample_role="sampler", island_gpus=1)
    assert "must divide" in full_parameter_layout_problem(
        data_parallel=1, actor_num_nodes=1, rollout_gpus=None, rollout_gpus_per_engine=2,
        sample_role="sampler", island_gpus=3)
    assert "must divide" in full_parameter_layout_problem(
        data_parallel=1, actor_num_nodes=1, rollout_gpus=None, rollout_gpus_per_engine=0,
        sample_role="sampler", island_gpus=2)


def test_full_parameter_sampler_island_translates_to_an_engine_only_miles_argv(tmp_path):
    """The island path up to the Miles argv: the S26 E5 one-card sampler was refused here."""
    from yeto.rl.adapters.miles.island_entry import build_ports_launch, parse_args
    from yeto.rl.engine.run_config import resolve_rl_run_config
    from yeto.rl.stage1 import expand, load_plan

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": _MINI_TEXT_CONFIG}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"messages": [{"role": "user", "content": "hi"}], "label": "1"}\n')
    argv = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"][3:]
    argv = [str(model) if a == "/data/models/glm5next-mini-bf16-7fb7883" else a for a in argv]
    argv = [str(REPO / a) if a.startswith("examples/rl_algorithms/") else a for a in argv]
    argv[argv.index("--rl-sample-role") + 1] = "sampler"
    args = parse_args(argv)
    config = resolve_rl_run_config(args, model_path=model, prompt_path=prompts,
                                   provider=glm5_next.glm5_next_provider(model), target_modules=[],
                                   yeto_policy_sync=False, verify_ref_load=False)
    assert config.parallel.dedicated_rollout_gpus is None
    assert config.parallel.visible_gpus_per_node == args.actor_num_gpus_per_node
    miles = list(build_ports_launch(args, config, []).argv)
    assert "--debug-rollout-only" in miles and "--debug-train-only" not in miles
    assert miles[miles.index("--num-gpus-per-node") + 1] == str(args.actor_num_gpus_per_node)


# ------------------------------------------------------------------ follower
def _store(tmp_path: Path) -> PolicyReleaseStore:
    return PolicyReleaseStore(CheckpointStore(str(tmp_path / "store")), keep=8)


def _release(store: PolicyReleaseStore, version: int, tag: str = "a"):
    return store.release(version, f"h{version}{tag}",
                         lambda d: (d / "model.safetensors").write_bytes(b"w" * (version + 1)),
                         weights=FULL)


class _Loads:
    def __init__(self) -> None:
        self.loaded: list[tuple[int, int]] = []

    def __call__(self, release, directory) -> None:
        assert (directory / "model.safetensors").is_file()
        self.loaded.append((release.seq, release.version))


def test_follower_loads_newest_only_and_counts_superseded(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads)
    _release(store, 1)
    assert follower.refresh().version == 1
    _release(store, 2)
    _release(store, 3)
    assert follower.refresh().version == 3 and follower.refresh() is None
    assert loads.loaded == [(1, 1), (3, 3)]
    assert follower.counters.to_dict() == {"loaded": 2, "superseded": 1, "stale_reads": 0,
                                           "lineage_resets": 0, "deferred_in_flight": 0}


def test_follower_never_goes_back_on_a_stale_store_read(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads)
    first = _release(store, 1)
    follower.refresh()
    second = _release(store, 2)
    follower.refresh()
    real_latest = store.latest
    store.latest = lambda: first  # a mount serving an old LATEST.json
    assert follower.refresh() is None
    assert follower.last_seen == second and follower.version == 2
    assert follower.counters.stale_reads == 1
    store.latest = real_latest
    assert loads.loaded == [(1, 1), (2, 2)]


def test_follower_loads_a_new_lineage_with_a_lower_version(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads)
    _release(store, 5)
    follower.refresh()
    _release(store, 3, tag="restored")  # a trainer restored from an older cut
    assert follower.refresh().version == 3
    assert follower.counters.lineage_resets == 1 and follower.version_lag() == 0


def test_follower_defers_updates_while_a_round_generates(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads)
    _release(store, 1)
    follower.refresh()
    _release(store, 2)
    with follower.generating():
        assert follower.refresh() is None
        assert follower.version == 1 and follower.version_lag() == 1
        with pytest.raises(RuntimeError, match="already generating"):
            with follower.generating():
                pass
    assert follower.counters.deferred_in_flight == 1
    assert follower.refresh().version == 2 and follower.version_lag() == 0


def test_follower_with_live_update_loads_during_a_round(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads, live_update=True)
    _release(store, 1)
    with follower.generating():
        assert follower.refresh().version == 1
    assert follower.counters.deferred_in_flight == 0


# ------------------------------------------------------------------ loop
class _Buffer:
    def __init__(self) -> None:
        self.groups = 0
        self.ended = False

    def refresh(self) -> None:
        pass

    def __len__(self) -> int:
        return self.groups

    def mark_data_end(self) -> None:
        self.ended = True


def test_sampler_loop_reports_round_version_lag_and_defers_mid_round_releases(tmp_path):
    store, loads = _store(tmp_path), _Loads()
    follower = PolicyFollower(store, loads)
    buffer, events = _Buffer(), []
    _release(store, 1)

    def generate(rollout_id: int) -> None:
        if rollout_id == 0:  # a release lands mid-round; a refresh here must not load it
            _release(store, 2)
            assert follower.refresh() is None
        buffer.groups += 1

    loop = SamplerLoop(follower=follower, buffer=buffer, generate=generate, num_rollouts=2,
                       max_backlog_groups=100, emit=lambda e, **f: events.append((e, f)),
                       sleep=lambda s: None)
    assert loop.run() == 2
    rounds = [f for e, f in events if e == "rl_sampler_round"]
    assert [(r["policy_version"], r["version_lag"]) for r in rounds] == [(1, 0), (2, 0)]
    assert rounds[-1]["releases"]["deferred_in_flight"] == 1
    assert loads.loaded == [(1, 1), (2, 2)] and buffer.ended


# ------------------------------------------------------------------ per-completion versions
def _completion(version: int | None = None) -> Completion:
    return Completion(token_ids=(1,), logprobs=(-0.1,), text="x", finish_reason="stop",
                      weight_version=version)


def test_completion_versions_follow_the_engine_capability():
    assert _completion_version(_completion(), 4, live=False) == 4
    assert _completion_version(_completion(4), 4, live=False) == 4
    with pytest.raises(SamplerEngineError, match="does not update"):
        _completion_version(_completion(5), 4, live=False)
    assert _completion_version(_completion(3), 4, live=True) == 3
    with pytest.raises(SamplerEngineError, match="weight_version"):
        _completion_version(_completion(), 4, live=True)


def test_engine_capability_is_opt_in():
    class Plain:
        pass

    class Live:
        updates_during_generation = True

    assert not updates_during_generation(Plain()) and updates_during_generation(Live())
