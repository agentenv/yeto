"""Synced trainers fed by external sampler pools that follow their own trainer
(``--rl-sampler-follow trainer`` with ``--rl-island-buffer-shards``, no sampler
island; dry run, no cloud, no Ray).

Every island is a synced trainer, learner k of K, and reads the sub-directory
``trainer<k>`` of the checkpoint store (its own lane) with the buffer shards of the
pools bound to it; the pools run outside the launch. Before, the sub-directory was
set only when the launch had sampler islands, and ``--rl-sampler-follow trainer``
needed them. A synced group with sampler islands keeps its sub-directories,
numbered by learner id also when the sampler islands come first."""

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import OWN_LANE

STORE = "modal-volume://yeto-fleet/run"
MOUNT = "/root/yeto-checkpoint-store/run"
TRAINERS = ("--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
            "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4",
            "--local-rl-rounds-per-sync", "2", "--tuning", "full",
            "--rl-sample-role", "trainer", "--rl-max-policy-age", "2",
            "--rl-sampler-follow", "trainer", "--rl-sample-wait-s", "900", "--rl-stall-timeout", "900",
            "--rl-checkpoint-store", STORE,
            "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json",
            "--cluster-prefix", "fleet-p1")
TWO = "modal:1xh100,modal:1xh100"
SHARDS = ("--rl-island-buffer-shards", "0:poolA-i0-0,poolA-i0-1",
          "--rl-island-buffer-shards", "1:poolB-i1-0")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-p1-test-key")


def _prepared(*flags, gpu=TWO, base=TRAINERS):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (*base, *flags), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _islands(args):
    """(plan, [(island request, parsed island args)]) of the dry-run plan; a
    trainer's ``$LEARNER_ID`` is its learner id."""
    from yeto.launcher import dry_run_plan

    plan = dry_run_plan(args)
    trainers = [i["learner_id"] for i in plan["island_requests"] if i.get("sample_role") == "trainer"]
    out = []
    for island in plan["island_requests"]:
        learner = trainers.index(island["learner_id"]) if island["learner_id"] in trainers else 0
        tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t)
                  for t in shlex.split(tail)]
        out.append((island, island_entry.parse_args(tokens)))
    return plan, out


# ------------------------------------------------ 1.1: every trainer gets trainer<k>
def test_k_trainers_and_a_syncer_without_sampler_islands_read_their_own_sub_store():
    args = _prepared(*SHARDS)
    assert sample_roles.is_external_pool_group(args) and sample_roles.follows_trainer(args)
    assert not sample_roles.is_synced_group(args)
    plan, islands = _islands(args)
    assert plan["outer_sync"]
    syncer = plan["role_syncers"][0]["command"]
    assert "--learners 2 " in syncer  # both islands are trainers behind the syncer
    shards = {0: ["poolA-i0-0", "poolA-i0-1"], 1: ["poolB-i1-0"]}
    for k, (request, trainer) in enumerate(islands):
        assert request["sample_role"] == "trainer"
        assert request["sample_substore"] == f"trainer{k}"
        assert request["buffer_shards"] == shards[k]
        command = request["learner_command"]
        assert f"--rl-sample-store {MOUNT}/trainer{k} --rl-sample-lanes 1" in command
        assert f"--rl-buffer-shards {','.join(shards[k])}" in command
        assert "--rl-island-buffer-shards" not in command  # a launch flag, never an island flag
        assert trainer.rl_sample_store == f"{MOUNT}/trainer{k}" and trainer.rl_sample_lanes == 1
        assert trainer.rl_buffer_shards == shards[k]
        assert sample_roles.buffer_lane(trainer) == OWN_LANE
        assert trainer.syncer == "10.0.0.1:29400"
        assert sample_roles.resolve(trainer, groups_per_round=4).store == f"{MOUNT}/trainer{k}"


def test_an_island_trainer_option_reaches_only_its_island():
    base = tuple("1" if a == "2" and TRAINERS[i - 1] == "--rl-max-policy-age" else a
                 for i, a in enumerate(TRAINERS))
    args = _prepared(*SHARDS, "--rl-island-release-every", "1:2", "--rl-island-max-policy-age", "1:2",
                     base=base)
    _plan, ((_r0, t0), (_r1, t1)) = _islands(args)
    assert t0.rl_max_policy_age == 1 and t1.rl_max_policy_age == 2
    assert sample_roles.resolve(t0, groups_per_round=4).release_every == 1
    assert sample_roles.resolve(t1, groups_per_round=4).release_every == 2


def test_without_the_flag_trainers_keep_the_shared_store_and_lanes():
    i = TRAINERS.index("--rl-sampler-follow")
    args = _prepared(base=TRAINERS[:i] + TRAINERS[i + 2:])
    _plan, islands = _islands(args)
    for request, trainer in islands:
        assert "sample_substore" not in request and "buffer_shards" not in request
        assert trainer.rl_sample_store == MOUNT and trainer.rl_sample_lanes == 2


# ------------------------------------------------ 1.2: the checks
def _args(**over):
    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=False, rl_island_sample_role=None,
                rl_sampler_follow="trainer", rl_backend="miles", rl_max_policy_age=2,
                rl_checkpoint_store="/store", rl_island_buffer_shards=["0:a", "1:b"],
                rl_island_release_every=None, rl_island_max_policy_age=None)
    return SimpleNamespace(**{**base, **over})


def test_external_pools_pass_the_follow_check():
    sample_roles.check_sampler_follow(_args(), 2)
    assert sample_roles.island_buffer_shards(_args(), 2) == {0: ["a"], 1: ["b"]}


def test_a_trainer_island_without_shards_is_named():
    with pytest.raises(ValueError, match=r"trainer island\(s\) \[1, 2\] name no buffer shard"):
        sample_roles.check_sampler_follow(_args(rl_island_buffer_shards=["0:a"]), 3)


@pytest.mark.parametrize(("entries", "match"), [
    (["0:a", "0:b"], "names island 0 twice"),
    (["0:a", "2:b"], "--gpu has 2 island"),
    (["x:a", "1:b"], "takes ISLAND:VALUE"),
    (["0:a", "1:"], "island 1 names no shard"),
    (["0:a", "1:a"], "shard 'a' is named for island 0 and island 1"),
    (["0:a b", "1:c"], "is not a shard name"),
])
def test_malformed_shard_entries_are_refused(entries, match):
    with pytest.raises(ValueError, match=match):
        sample_roles.check_sampler_follow(_args(rl_island_buffer_shards=entries), 2)


def test_release_every_above_the_island_policy_age_is_refused_for_that_island():
    sample_roles.check_sampler_follow(_args(rl_island_release_every=["1:2"], rl_island_max_policy_age=["1:2"]), 2)
    with pytest.raises(ValueError, match=r"trainer island 1: .*--rl-max-policy-age >= 3"):
        sample_roles.check_sampler_follow(_args(rl_island_release_every=["1:3"]), 2)
    with pytest.raises(ValueError, match=r"trainer island 0: .*--rl-max-policy-age >= 2"):
        sample_roles.check_sampler_follow(_args(rl_island_release_every=["0:2"],
                                                rl_island_max_policy_age=["0:1"]), 2)
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        sample_roles.check_sampler_follow(_args(rl_island_max_policy_age=["0:0"]), 2)


@pytest.mark.parametrize(("over", "match"), [
    ({"rl_island_buffer_shards": None, "rl_island_release_every": ["0:2"]},
     "need --rl-island-buffer-shards"),
    ({"rl_island_sample_role": ["1:sampler"]}, "no --rl-island-sample-role"),
    ({"rl_single_island_no_sync": True}, "with a syncer"),
    ({"rl_sampler_follow": "merged"}, "--rl-sampler-follow trainer"),
    ({"rl_buffer_shards": ["a", "b"]}, "take each trainer's shards from --rl-island-buffer-shards"),
])
def test_the_external_pool_flags_belong_to_synced_trainers_that_follow(over, match):
    with pytest.raises(ValueError, match=match):
        sample_roles.check_sampler_follow(_args(**over), 2)


def test_follow_trainer_without_sampler_islands_or_pools_is_still_refused():
    with pytest.raises(ValueError, match="needs a synced group"):
        sample_roles.check_sampler_follow(_args(rl_island_buffer_shards=None), 2)
    with pytest.raises(ValueError, match="--rl-backend miles only"):
        sample_roles.check_sampler_follow(_args(rl_backend="verl"), 2)


def test_the_launch_refuses_a_missing_island_before_any_cloud_work():
    with pytest.raises(ValueError, match=r"trainer island\(s\) \[1\] name no buffer shard"):
        _prepared("--rl-island-buffer-shards", "0:poolA-i0-0")


def test_an_island_policy_age_above_the_algorithm_staleness_is_refused():
    from yeto.launch_preflight import PreflightError, check_policy_age_spec

    args = _prepared(*SHARDS, "--rl-island-max-policy-age", "1:2")
    check_policy_age_spec(args)
    args.rl_island_max_policy_age = ["1:64"]
    with pytest.raises(PreflightError, match="trainer island 1: "):
        check_policy_age_spec(args)


# ------------------------------------------------ 1.3: sub-store number = learner id
def test_sampler_islands_first_trainers_after_number_sub_stores_by_learner_id():
    flags = ("--rl-island-sample-role", "0:sampler", "--rl-island-sample-role", "1:sampler")
    args = _prepared(*flags, gpu="modal:1xl40s,modal:1xl40s,modal:1xh100,modal:1xh100")
    assert sample_roles.synced_group_learners(args, 4) == {2: 0, 3: 1}
    assert sample_roles.sampler_bindings(args, 4) == {0: 2, 1: 3}
    assert [sample_roles.island_substore(args, m, 4) for m in range(4)] == \
        ["trainer0", "trainer1", "trainer0", "trainer1"]
    _plan, islands = _islands(args)
    (r0, s0), (r1, s1), (r2, t0), (r3, t1) = islands
    assert [r["sample_role"] for r in (r0, r1, r2, r3)] == ["sampler", "sampler", "trainer", "trainer"]
    # trainer island 2 is learner 0: trainer0/, with its sampler (island 0); island 3 is learner 1
    assert t0.rl_sample_store == f"{MOUNT}/trainer0" and t1.rl_sample_store == f"{MOUNT}/trainer1"
    assert s0.rl_resume_store == f"{MOUNT}/trainer0" and s1.rl_resume_store == f"{MOUNT}/trainer1"
    assert [r["sample_substore"] for r in (r0, r1, r2, r3)] == ["trainer0", "trainer1", "trainer0", "trainer1"]
    assert "buffer_shards" not in r2  # its sampler island writes the index itself
