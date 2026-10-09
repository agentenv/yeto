"""agentic-rollout-utilization group 2: the policy-age limit switch (default 0)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.identity import backend_identity
from yeto.rl.engine.policy_age import (
    PolicyAgeError,
    PolicyAgeSupport,
    bind_policy_age,
    check_batch_ages,
    split_by_age,
    token_version,
)

MILES = backend_identity("ports").sha256()


# ---------------------------------------------------------------- 2.1 contract


def test_limit_zero_keeps_identity_and_nonzero_changes_it():
    assert bind_policy_age(MILES, 0) == MILES
    assert bind_policy_age(None, 3) is None
    one, two = bind_policy_age(MILES, 1), bind_policy_age(MILES, 2)
    assert len({MILES, one, two}) == 3 and len(one) == 64
    with pytest.raises(PolicyAgeError):
        bind_policy_age(MILES, -1)


def test_strict_handshake_refuses_an_island_with_another_limit():
    """HELLO session contract (strict / decoupled syncer): byte equality decides."""
    from tests.test_rl_backend_identity import _layout
    from yeto.protocol import layout_fingerprint
    from yeto.rl.engine.backend_identity import session_contract_hash

    fp = layout_fingerprint(_layout())
    syncer = session_contract_hash(fp, bind_policy_age(MILES, 0))
    assert session_contract_hash(fp, bind_policy_age(MILES, 0)) == syncer  # same limit: admitted
    assert session_contract_hash(fp, bind_policy_age(MILES, 1)) != syncer  # limit 1: refused


def test_elastic_join_carries_the_bound_identity(monkeypatch):
    """Elastic JOIN: the island identity bytes differ, so only that connection is refused."""
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.elastic_client import ElasticClientConfig

    zero = island_entry._backend_identity_sha256(SimpleNamespace(rl_engine="ports"))
    one = island_entry._backend_identity_sha256(SimpleNamespace(rl_engine="ports", rl_max_policy_age=1))
    assert zero == MILES and one == bind_policy_age(MILES, 1)
    a = ElasticClientConfig(("x", 0), 1, backend_identity_sha256=zero).backend_identity_bytes()
    b = ElasticClientConfig(("x", 0), 2, backend_identity_sha256=one).backend_identity_bytes()
    assert a != b


@pytest.mark.parametrize("backend", ["miles", "verl"])
def test_launcher_refuses_an_unsupported_limit_before_launch(backend):
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args(["--rl-max-policy-age", "1"])
    args.rl_backend = backend
    with pytest.raises(PolicyAgeError, match=f"backend '{backend}' supports up to stage 1"):
        _prepare_rl_args(args)


def test_launcher_default_limit_adds_no_island_flag():
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args()
    _prepare_rl_args(args)
    assert args.rl_max_policy_age == 0


def test_support_message_names_the_stage():
    support = PolicyAgeSupport("x", stage=2, max_policy_age=1)
    support.check(1)
    with pytest.raises(PolicyAgeError, match="stage 2"):
        support.check(2)


# ------------------------------------------------- 2.3 profile / driver checks


def test_profile_limit_needs_the_bounded_staleness_contract():
    from yeto.rl.engine.execution_profile import ExecutionProfile, ProfileError

    base = dict(name="p", execution_mode="colocated-serial", outer_protocol="strict-avg")
    assert ExecutionProfile(**base).max_policy_age == 0
    with pytest.raises(ProfileError, match="on-policy"):
        ExecutionProfile(**base, max_policy_age=1)
    with pytest.raises(ProfileError, match="needs max_policy_age > 0"):
        ExecutionProfile(**base, algorithm_contract="bounded-staleness")
    p = ExecutionProfile(**base, max_policy_age=1, algorithm_contract="bounded-staleness")
    assert p.contract_hash != ExecutionProfile(**base).contract_hash
    # limit 0 contract hash is the pre-change one (no new field in to_dict)
    assert set(ExecutionProfile(**base).to_dict()) == {
        "name", "execution_mode", "outer_protocol", "algorithm_contract", "max_policy_age",
        "groups_per_batch", "samples_per_group", "optimizer_steps_per_round",
        "max_inflight_batches", "ready_buffer_groups", "allowed_overlap", "publish_rule",
        "algorithm_spec_sha256", "lr_schedule_sha256", "schema"}


def _g(gid, version, versions=None, digest="h"):
    from yeto.rl.engine.ports import GroupMetadata

    return GroupMetadata(gid, (f"{gid}-s0",), f"yeto:{version}:{digest}{version}", 0.5, 0.5, 3,
                         policy_versions=versions)


def test_split_by_age_zero_one_and_over_limit():
    groups = [_g("a", 5), _g("b", 4), _g("c", 5, versions=(4, 5)), _g("d", 5, versions=(3, 4, 5))]
    kept, dropped = split_by_age(groups, 5, 0)
    assert [g.group_id for g in kept] == ["a"]
    kept, dropped = split_by_age(groups, 5, 1)
    assert [g.group_id for g in kept] == ["a", "b", "c"] and [g.group_id for g in dropped] == ["d"]
    assert token_version("yeto:7:abc") == 7 and token_version("bad") is None
    known = {3: "h3", 4: "h4", 5: "h5"}
    assert check_batch_ages(groups[:3], 5, 1, known) == []
    assert "exceeds max_policy_age=1" in check_batch_ages(groups[3:], 5, 1, known)[0]
    assert "not a published policy" in check_batch_ages([_g("e", 4, digest="x")], 5, 1, known)[0]


def _driver(tmp_path, limit, stale_group_rounds=(), versions=None):
    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, stale_token_rounds=set(stale_group_rounds))
    if versions is not None:
        import dataclasses

        original = engine.rollout.generate

        def gen(rid, **kw):
            h = original(rid, **kw)
            groups = (dataclasses.replace(h.groups[0], policy_versions=versions(rid)),) + h.groups[1:]
            return dataclasses.replace(h, groups=groups)

        engine.rollout.generate = gen
    import dataclasses

    from yeto.rl.engine.algorithm import CorrectionSpec, ExecutionSpec
    from yeto.rl.engine.execution_profile import ExecutionProfile

    algorithm, caps, profile = AlgorithmSpec(), fake_capabilities(), None
    if limit:
        algorithm = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=limit),
                                  correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
        caps = dataclasses.replace(caps, execution=dataclasses.replace(
            caps.execution, max_policy_staleness=limit))
        profile = ExecutionProfile("p", "colocated-serial", "strict-avg", max_policy_age=limit,
                                   algorithm_contract="bounded-staleness",
                                   groups_per_batch=engine.groups,
                                   samples_per_group=engine.samples_per_group,
                                   algorithm_spec_sha256=algorithm.sha256())
    return IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement, capabilities=caps,
                        algorithm=algorithm, sync=LocalOnlySync(3), profile=profile,
                        events=EventTape(tmp_path / "e.jsonl", 0))


def test_driver_limit_zero_keeps_refusing_older_tokens(tmp_path):
    from yeto.rl.engine.driver import PolicyIdentityError

    with pytest.raises(PolicyIdentityError, match="not sampled from"):
        _driver(tmp_path, 0, stale_group_rounds={1}).run()
    with pytest.raises(PolicyIdentityError, match="older policy versions"):
        _driver(tmp_path, 0, versions=lambda rid: (max(0, rid - 1), rid)).run()


def test_driver_limit_one_accepts_version_segments_within_the_window(tmp_path):
    """Limit 1: a group whose tokens span v-1 and v trains in round v."""
    _driver(tmp_path, 1, versions=lambda rid: (max(0, rid - 1), rid)).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["event"] == "rl_round_trained"]) == 3


def test_driver_limit_one_refuses_a_group_two_versions_old(tmp_path):
    from yeto.rl.engine.driver import PolicyIdentityError

    with pytest.raises(PolicyIdentityError, match="exceeds max_policy_age=1"):
        _driver(tmp_path, 1, versions=lambda rid: (max(0, rid - 2), rid)).run()


# ------------------------------------------------------ 2.4 Miles derivation


def test_miles_switches_are_derived_from_the_limit():
    from yeto.rl.adapters.miles import algorithm_flags as af
    from yeto.rl.adapters.miles.policy_age import policy_age_argv
    from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, ExecutionSpec

    assert policy_age_argv(0) == ()
    assert policy_age_argv(1) == ("--partial-rollout", "--mask-offpolicy-in-partial-rollout")
    assert "--partial-rollout" in af.mapped_flags()
    assert "--partial-rollout" not in af.UNMAPPED_OBJECTIVE_FLAGS
    row = af.MAPPINGS["--partial-rollout"]
    assert row.translate(AlgorithmSpec()) == []
    assert row.translate(AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1))) == [
        "--partial-rollout", "--mask-offpolicy-in-partial-rollout"]
    with pytest.raises((AlgorithmSpecError, ValueError), match="rl-max-policy-age"):
        af.absorb_extra_argv(AlgorithmSpec(), ["--partial-rollout"])
