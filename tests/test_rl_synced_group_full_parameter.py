"""S27 G-BG: a full-parameter synced group on Miles (dry run; no cloud, no Ray).

Two trainer islands merged by the syncer (``--rl-sample-role trainer``,
full-parameter asynchronous rounds) and two sampler islands of the same launch
(``--rl-island-sample-role M:sampler``). Checked: the launch passes the
full-parameter, placement and sampler-option rules; every island command
parses on the Miles island entry; with ``--rl-sampler-follow trainer`` each
trainer and its own sampler share one store sub-directory, and the trainer reads
its own lane and releases its local versions."""

from __future__ import annotations

import shlex

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import OWN_LANE, BufferLane, release_version

GROUP_GPU = "modal:8xh200,modal:8xh200,modal:8xh100,modal:8xh100"
STORE = "modal-volume://yeto-gbg/run"
MOUNT = "/root/yeto-checkpoint-store/run"
GROUP = ("--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
         "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4",
         "--local-rl-rounds-per-sync", "2",
         "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler",
         "--rl-island-sample-role", "3:sampler", "--rl-max-policy-age", "2",
         "--rl-sampler-max-backlog-rounds", "1", "--rl-sample-wait-s", "900",
         "--rl-checkpoint-store", STORE, "--rollout-num-gpus-per-engine", "2",
         "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json",
         "--cluster-prefix", "gbg-group")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "gbg-test-key")


def _plan(extra=()):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", (*GROUP, *extra), gpu=GROUP_GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _islands(plan):
    """[(island request, parsed island args)], with the learner id each island runs as."""
    out = []
    trainers = [i["learner_id"] for i in plan["island_requests"] if i["sample_role"] == "trainer"]
    for island in plan["island_requests"]:
        command = island["learner_command"]
        learner = (trainers.index(island["learner_id"]) if island["learner_id"] in trainers else 0)
        tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t)
                  for t in shlex.split(tail)]
        out.append((island, island_entry.parse_args(tokens)))
    return out


def test_group_launch_passes_without_a_fixed_partition():
    _, plan = _plan()
    assert plan["outer_sync"] and plan["total_gpus"] == 32
    assert [i["sample_role"] for i in plan["island_requests"]] == ["trainer", "trainer",
                                                                     "sampler", "sampler"]
    assert all(i["placement"]["placement"] == "colocated" for i in plan["island_requests"])


def test_group_refuses_a_fixed_partition_on_store_fed_trainers():
    with pytest.raises((ValueError, SystemExit), match="trainers start no engine"):
        _plan(("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "4"))


def test_merged_follow_islands_parse_and_share_one_store():
    _, plan = _plan()
    (_, t0), (_, t1), (_, s0), (_, s1) = _islands(plan)
    for k, trainer in enumerate((t0, t1)):
        assert trainer.rl_sample_role == "trainer" and trainer.sync_preset == "full-parameter"
        assert trainer.parameter_mode == "full" and trainer.rl_sample_lanes == 2
        assert trainer.rl_sample_store == MOUNT
        assert sample_roles.buffer_lane(trainer) == BufferLane(k, 2)
        assert trainer.rl_sampler_max_backlog_rounds is None
    for j, sampler in enumerate((s0, s1)):
        # a pair's full-parameter sampler: no sync, full tuning, its backlog cap and slot
        assert sampler.rl_single_island_no_sync and sampler.sync_preset == "strict-avg"
        assert sampler.parameter_mode == "full" and sampler.fragments == 1
        assert sampler.rl_resume_store == MOUNT and sampler.learner_id == 0
        assert sampler.rl_sampler_max_backlog_rounds == 1
        assert sampler.rl_sampler_data_slot == f"{j}/2"
        assert sampler.rl_buffer_shards == ["island2", "island3"]
        assert sampler.global_rounds == 3  # 2 trainers x 3 rounds over 2 samplers


def test_trainer_follow_binds_each_sampler_to_one_trainer_sub_store():
    args, plan = _plan(("--rl-sampler-follow", "trainer"))
    assert sample_roles.sampler_bindings(args, 4) == {2: 0, 3: 1}
    (_, t0), (_, t1), (_, s0), (_, s1) = _islands(plan)
    for k, (trainer, sampler) in enumerate(((t0, s0), (t1, s1))):
        substore = f"{MOUNT}/trainer{k}"
        assert trainer.rl_sample_store == substore and trainer.rl_sample_lanes == 1
        assert sample_roles.buffer_lane(trainer) == OWN_LANE
        assert sampler.rl_resume_store == substore
        assert sampler.rl_buffer_shards is None  # one sampler per trainer writes the index
        assert sampler.rl_sampler_data_slot == f"{k}/2"  # disjoint prompts over all samplers
        assert sampler.global_rounds == 3  # its trainer's 3 rounds
    assert t0.syncer == t1.syncer == "10.0.0.1:29400"


def test_trainer_follow_with_two_samplers_per_trainer_shards_its_sub_store():
    gpu = "modal:8xh200,modal:8xh200,modal:8xh100,modal:8xh100,modal:8xh100,modal:8xh100"
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", (*GROUP, "--rl-island-sample-role", "4:sampler",
                                    "--rl-island-sample-role", "5:sampler",
                                    "--rl-sampler-follow", "trainer", "--total-steps", "4"),
                          gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    assert sample_roles.sampler_bindings(args, 6) == {2: 0, 3: 1, 4: 0, 5: 1}
    islands = _islands(dry_run_plan(args))
    samplers = [parsed for _, parsed in islands[2:]]
    assert [s.rl_resume_store for s in samplers] == [f"{MOUNT}/trainer{k}" for k in (0, 1, 0, 1)]
    assert samplers[0].rl_buffer_shards == samplers[2].rl_buffer_shards == ["island2", "island4"]
    assert islands[0][1].rl_buffer_shards == ["island2", "island4"]
    assert [s.global_rounds for s in samplers] == [2, 2, 2, 2]  # 2 x 4 rounds over 4 samplers


def test_trainer_follow_refuses_samplers_that_do_not_split_over_the_trainers():
    from yeto.launcher import _prepare_rl_args

    # 3 samplers share 2 x 3 rounds (the group rule holds) but not 2 trainers evenly
    args = _launcher_args("ports", (*GROUP, "--rl-island-sample-role", "4:sampler",
                                    "--rl-sampler-follow", "trainer"), gpu=GROUP_GPU + ",modal:8xh100")
    args.controller = "local"
    with pytest.raises((ValueError, SystemExit), match="do not split evenly"):
        _prepare_rl_args(args)


def test_trainer_follow_needs_a_synced_group():
    with pytest.raises(ValueError, match="needs a synced group"):
        sample_roles.check_sampler_follow(
            _ns(rl_sampler_follow="trainer", rl_island_sample_role=None, rl_sample_role="trainer"), 2)


def _ns(**kw):
    from types import SimpleNamespace

    base = dict(rl_single_island_no_sync=False, rl_node_class=None)
    return SimpleNamespace(**{**base, **kw})


def test_own_lane_releases_the_local_version():
    from types import SimpleNamespace

    state = SimpleNamespace(policy_version=5)
    driver = SimpleNamespace(outer_version=2)
    assert release_version(driver, state, OWN_LANE) == 5
    assert release_version(driver, state, BufferLane(1, 2)) == 2
    assert BufferLane.parse("0/1") == OWN_LANE and OWN_LANE.is_own


def test_island_accepts_one_lane_only_on_a_synced_trainer():
    island_entry._check_sample_store(_ns(rl_sample_role="trainer", rl_sample_store="/s",
                                         rl_sample_lanes=1, learner_id=1, rl_resume_store=None))
    with pytest.raises(ValueError, match="--rl-sample-lanes >= 1"):
        island_entry._check_sample_store(_ns(rl_sample_role="trainer", rl_sample_store="/s",
                                             rl_sample_lanes=0, learner_id=0, rl_resume_store=None))


def test_group_forwards_delta_releases_to_its_trainers_only():
    _, plan = _plan(("--rl-sampler-follow", "trainer", "--rl-release-delta-full-every", "8"))
    (_, t0), (_, t1), (_, s0), (_, s1) = _islands(plan)
    assert t0.rl_release_delta_full_every == t1.rl_release_delta_full_every == 8
    assert s0.rl_release_delta_full_every == s1.rl_release_delta_full_every == 0
    config = sample_roles.resolve(t1, groups_per_round=4)
    assert config.release_delta_full_every == 8 and config.store == f"{MOUNT}/trainer1"
