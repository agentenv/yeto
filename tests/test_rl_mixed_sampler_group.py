"""Sampler islands and external pools in one synced group (a *mixed group*:
``--rl-island-sample-role M:sampler`` with ``--rl-island-buffer-shards K:SHARDS``
and ``--rl-sampler-follow trainer``; dry run, no cloud, no Ray).

A trainer reads the shards of the sampler islands bound to it (each writes the
shard of its node name, ``island<M>``) and the shards of its external pools, all
in its sub-directory ``trainer<k>``; K of the external-pool flags is the
trainer's learner id. The pure synced group and the pure external-pool launch
keep their plans."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_external_pool_follow import (MOUNT, SHARDS, TWO, _age_base, _identities, _islands, _prepared,
                                         _session)
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import OWN_LANE


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-mixed-test-key")


# sampler island 0 first, then two trainers (islands 1, 2 = learners 0, 1); the
# sampler feeds learner 1, learner 0 has pools only
THREE = "modal:1xl40s,modal:1xh100,modal:1xh100"
MIXED = ("--rl-island-sample-role", "0:sampler",
         "--rl-island-buffer-shards", "0:poolA-i0-0",
         "--rl-island-buffer-shards", "1:poolB-i1-0,poolB-i1-1",
         "--rl-island-sampler-follows", "0:1",
         "--rl-island-release-every", "1:2", "--rl-island-max-policy-age", "1:2")


def test_a_trainer_reads_its_sampler_islands_and_its_pools_in_its_own_sub_store():
    args = _prepared(*MIXED, gpu=THREE)
    assert sample_roles.is_mixed_group(args) and sample_roles.is_synced_group(args)
    assert not sample_roles.is_external_pool_group(args)
    assert sample_roles.synced_group_learners(args, 3) == {1: 0, 2: 1}
    assert sample_roles.sampler_bindings(args, 3) == {0: 2}  # sampler island 0 -> learner 1 (island 2)
    plan, islands = _islands(args)
    assert "--learners 2 " in plan["role_syncers"][0]["command"]  # the sampler is not a learner
    (r0, sampler), (r1, t0), (r2, t1) = islands
    assert [r["sample_role"] for r in (r0, r1, r2)] == ["sampler", "trainer", "trainer"]
    assert [r.get("sample_substore") for r in (r0, r1, r2)] == ["trainer1", "trainer0", "trainer1"]
    assert r0["feeds_trainer"] == 2 and r0["sampler_until_trainers_end"] == 1
    # learner 0: its pools only; learner 1: its sampler island's shard, then its pools'
    assert r1["buffer_shards"] == ["poolA-i0-0"] and "buffer_shards" not in r0
    assert r2["buffer_shards"] == ["island0", "poolB-i1-0", "poolB-i1-1"]
    assert t0.rl_sample_store == f"{MOUNT}/trainer0" and t1.rl_sample_store == f"{MOUNT}/trainer1"
    assert t0.rl_sample_lanes == 1 and t1.rl_sample_lanes == 1
    assert sample_roles.buffer_lane(t0) == OWN_LANE and sample_roles.buffer_lane(t1) == OWN_LANE
    assert t0.rl_buffer_shards == ["poolA-i0-0"]
    assert t1.rl_buffer_shards == ["island0", "poolB-i1-0", "poolB-i1-1"]
    assert int(t0.rl_release_every or 1) == 1 and int(t1.rl_release_every) == 2  # K is the learner id
    # the sampler island writes its own shard in its trainer's sub-store, until that trainer ended
    assert sampler.rl_resume_store == f"{MOUNT}/trainer1"
    assert sampler.rl_sampler_node == "island0" and "island0" in sampler.rl_buffer_shards
    assert sampler.rl_sampler_until_trainers_end == 1
    for request in (r0, r1, r2):
        for flag in ("--rl-island-buffer-shards", "--rl-island-sampler-follows", "--rl-island-release-every"):
            assert flag not in request["learner_command"]  # launch flags, never island flags


def test_without_the_binding_flag_sampler_j_feeds_trainer_j_mod_k():
    flags = ("--rl-island-sample-role", "2:sampler", "--rl-island-sample-role", "3:sampler",
             "--rl-island-buffer-shards", "0:poolA-i0-0", "--rl-island-buffer-shards", "1:poolB-i1-0")
    args = _prepared(*flags, gpu=f"{TWO},modal:1xl40s,modal:1xl40s")
    assert sample_roles.sampler_bindings(args, 4) == {2: 0, 3: 1}
    _plan, islands = _islands(args)
    assert [r["buffer_shards"] for r, _ in islands[:2]] == [["island2", "poolA-i0-0"], ["island3", "poolB-i1-0"]]
    assert [s.rl_resume_store for _, s in islands[2:]] == [f"{MOUNT}/trainer0", f"{MOUNT}/trainer1"]


def test_one_full_parameter_trainer_with_a_sampler_island_and_a_pool():
    """A fleet of one island: trainer + Miles sampler + one external pool."""
    flags = ("--rl-island-sample-role", "1:sampler", "--rl-island-buffer-shards", "0:poolA-i0-0",
             "--rl-island-sampler-follows", "1:0")
    args = _prepared(*flags, gpu="modal:1xh100,modal:1xl40s")
    plan, islands = _islands(args)
    assert "--learners 1 " in plan["role_syncers"][0]["command"]
    (rt, trainer), (rs, sampler) = islands
    assert rt["buffer_shards"] == ["island1", "poolA-i0-0"] and trainer.rl_sample_store == f"{MOUNT}/trainer0"
    assert sampler.rl_resume_store == f"{MOUNT}/trainer0" and rs["sample_substore"] == "trainer0"


def test_the_pure_forms_keep_their_plans():
    """No sampler island: K is the island index and the shards are the pools' only; no
    external pool: a trainer with one sampler island has no shard list (the sampler
    writes the index itself)."""
    _plan, pure_pools = _islands(_prepared(*SHARDS))
    assert [r["buffer_shards"] for r, _ in pure_pools] == [["poolA-i0-0", "poolA-i0-1"], ["poolB-i1-0"]]
    flags = ("--rl-island-sample-role", "0:sampler", "--rl-island-sample-role", "1:sampler")
    args = _prepared(*flags, gpu="modal:1xl40s,modal:1xl40s,modal:1xh100,modal:1xh100")
    assert not sample_roles.is_mixed_group(args)
    _plan, b1 = _islands(args)
    assert all("buffer_shards" not in r for r, _ in b1)
    assert [i.rl_buffer_shards for _, i in b1] == [None] * 4


# ------------------------------------------------ the checks
def _args(**over):
    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=False,
                rl_island_sample_role=["0:sampler"], rl_sampler_follow="trainer", rl_backend="miles",
                rl_max_policy_age=2, rl_checkpoint_store="/store", tuning="full",
                rl_sync_preset="full-parameter", rl_sync_full_parameter_rounds="async",
                rl_island_buffer_shards=["0:a", "1:b"], rl_island_sampler_follows=["0:1"],
                rl_island_release_every=None, rl_island_max_policy_age=None, total_steps=4)
    return SimpleNamespace(**{**base, **over})


def test_a_mixed_group_passes_the_follow_check():
    sample_roles.check_sampler_follow(_args(), 3)
    assert sample_roles.island_buffer_shards(_args(), 3) == {0: ["a"], 1: ["b"]}  # keyed by learner id


@pytest.mark.parametrize(("over", "match"), [
    ({"rl_island_buffer_shards": ["0:a", "2:b"]}, "learner id; the launch has 2 trainer"),
    ({"rl_island_buffer_shards": ["0:a", "1:island0"]}, "'island0' of trainer 1 is the shard sampler island 0"),
    ({"rl_island_buffer_shards": ["0:a", "1:a"]}, "shard 'a' is named for island 0 and island 1"),
    ({"rl_island_buffer_shards": ["0:a"], "rl_island_sampler_follows": ["0:0"]},
     r"learner id \[1\] have no sampler"),
    ({"rl_island_sampler_follows": ["1:0"]}, "island 1 is not a sampler island"),
    ({"rl_island_sampler_follows": ["0:2"]}, "K must be a trainer's learner id, 0..1"),
    ({"rl_island_sample_role": ["0:sampler", "3:sampler"]}, "name every sampler island once"),
    ({"rl_buffer_shards": ["a"]}, "--rl-buffer-shards: a synced group with external sampler pools"),
    ({"rl_sync_preset": "strict-avg", "tuning": "lora"}, "full-parameter or decoupled"),
    ({"rl_island_release_every": ["2:2"]}, "learner id; the launch has 2 trainer"),
])
def test_bad_mixed_groups_are_refused(over, match):
    with pytest.raises(ValueError, match=match):
        sample_roles.check_sampler_follow(_args(**over), 4 if "3:sampler" in str(over) else 3)


def test_the_binding_flag_belongs_to_a_group_whose_samplers_follow_their_trainer():
    with pytest.raises(ValueError, match="--rl-island-sampler-follows binds the sampler islands"):
        sample_roles.check_sampler_follow(_args(rl_island_sample_role=None), 2)
    with pytest.raises(ValueError, match="--rl-island-sampler-follows binds the sampler islands"):
        sample_roles.check_sampler_follow(_args(rl_sampler_follow="merged"), 3)


def test_the_binding_flag_alone_binds_sampler_islands_without_pools():
    """Islands with Miles samplers only (a fleet without external pools): each
    trainer reads its own sampler islands' shards, every trainer needs one."""
    args = _args(rl_island_sample_role=["0:sampler", "3:sampler"], rl_island_buffer_shards=None,
                 rl_island_sampler_follows=["0:1", "3:0"])
    assert sample_roles.is_mixed_group(args)
    sample_roles.check_sampler_follow(args, 4)
    assert sample_roles.sampler_bindings(args, 4) == {0: 2, 3: 1}
    assert sample_roles.mixed_trainer_shards(args, 1, 4) == ["island3"]
    assert sample_roles.mixed_trainer_shards(args, 2, 4) == ["island0"]
    with pytest.raises(ValueError, match=r"learner id \[0\] have no sampler"):
        sample_roles.check_sampler_follow(_args(rl_island_buffer_shards=None), 3)


def test_the_launch_refuses_a_bad_mixed_group_before_any_cloud_work():
    with pytest.raises(ValueError, match=r"learner id \[0\] have no sampler"):
        _prepared("--rl-island-sample-role", "0:sampler", "--rl-island-buffer-shards", "1:poolB-i1-0",
                  "--rl-island-sampler-follows", "0:1", gpu=THREE)


def test_an_island_policy_age_of_a_mixed_group_names_the_learner():
    from yeto.launch_preflight import PreflightError, check_policy_age_spec

    args = _prepared(*MIXED, gpu=THREE)
    check_policy_age_spec(args)
    args.rl_island_max_policy_age = ["1:64"]
    with pytest.raises(PreflightError, match="trainer island 1: "):
        check_policy_age_spec(args)


def test_trainers_of_a_mixed_group_with_different_policy_ages_share_one_session_contract():
    """Learner 0 runs the launch limit 1, learner 1 its own 2: both bind the launch
    limit, so the syncer admits both; the sampler island does not sync."""
    from yeto.rl.adapters.miles import island_entry

    args = _prepared(*MIXED, gpu=THREE, base=_age_base(1))
    plan, _ = _islands(args)
    (sampler, _sid), (t0, id0), (t1, id1) = _identities(args)
    assert (t0.rl_max_policy_age, t1.rl_max_policy_age) == (1, 2)
    assert t0.rl_contract_max_policy_age == t1.rl_contract_max_policy_age == 1
    assert getattr(sampler, "rl_contract_max_policy_age", None) is None
    commands = [f"{r['learner_command']} " for r in plan["island_requests"]]
    assert "--rl-contract-max-policy-age" not in commands[0]
    assert all(" --rl-contract-max-policy-age 1 " in c for c in commands[1:])
    assert id0 == id1 == island_entry._backend_identity_sha256(
        SimpleNamespace(rl_engine="ports", rl_max_policy_age=1))
    assert _session(id0) == _session(id1)
