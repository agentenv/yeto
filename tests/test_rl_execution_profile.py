"""rl-infra-spec task 1.4: ExecutionProfile / readiness (pure, no GPU)."""

from __future__ import annotations

import pytest

from yeto.rl.elastic_benchmark.manifest import example_manifest
from yeto.rl.engine.execution_profile import (
    ExecutionProfile,
    ProfileError,
    ReadinessError,
    ReadinessSnapshot,
    dependency_table,
    generate_blockers,
    overlap_violation,
    quiescent_cut_blockers,
    require,
    train_blockers,
)


def _profile(mode="partitioned-serial", outer="strict-avg", **kw) -> ExecutionProfile:
    kw.setdefault("groups_per_batch", 4)
    kw.setdefault("samples_per_group", 8)
    return ExecutionProfile(name=f"p-{mode}", execution_mode=mode, outer_protocol=outer, **kw)


def _snap(**kw) -> ReadinessSnapshot:
    base = dict(
        rollout_id=3,
        optimizer_step=3,
        trained_policy_version=3,
        published_policy_version=3,
        publication_complete=True,
        driver_safe_point=True,
    )
    base.update(kw)
    return ReadinessSnapshot(**base)


def test_dependency_table_distinguishes_the_three_modes():
    serial = {r["task"]: r for r in dependency_table(_profile("colocated-serial"))}
    part = {r["task"]: r for r in dependency_table(_profile("partitioned-serial"))}
    over = {
        r["task"]: r
        for r in dependency_table(
            _profile("partitioned-overlap", allowed_overlap={("reward", "checkpoint")})
        )
    }
    # 2.3: at age 0 a second batch can never be in flight (queue bound).
    with pytest.raises(ProfileError, match="exactly one batch"):
        _profile("partitioned-overlap", allowed_overlap={("reward", "checkpoint")},
                 max_inflight_batches=2)
    # Same algorithm dependencies in every mode.
    for table in (serial, part, over):
        assert table["generate"]["depends_on_previous_round"] == ["publish[r-1]"]
        assert table["train"]["depends_on"] == ["reward"]
    assert serial["train"]["role"] == "shared-pool" and serial["train"]["serialized_by_resource"]
    assert part["train"]["role"] == "trainer" and part["generate"]["role"] == "rollout"
    assert not part["train"]["serialized_by_resource"]
    assert all(not r["may_overlap_with"] for r in part.values())
    assert over["reward"]["may_overlap_with"] == ["checkpoint"]


def test_strict_profile_never_generates_on_an_older_policy():
    p = _profile("partitioned-serial")
    assert generate_blockers(p, _snap()) == []
    stale = _snap(trained_policy_version=4, published_policy_version=3)
    reasons = generate_blockers(p, stale)
    assert any("behind trained" in r for r in reasons)
    with pytest.raises(ReadinessError):
        require(reasons, "generate")
    assert generate_blockers(p, _snap(publication_complete=False))


def test_overlap_that_would_let_generation_run_ahead_is_rejected():
    with pytest.raises(ProfileError, match="age 0"):
        _profile("partitioned-overlap", allowed_overlap={("generate", "train")})
    with pytest.raises(ProfileError, match="dependency"):
        _profile("partitioned-overlap", allowed_overlap={("train", "outer_sync")})
    with pytest.raises(ProfileError, match="trainer/trainer GPU role"):
        _profile("partitioned-overlap", allowed_overlap={("checkpoint", "train")})
    with pytest.raises(ProfileError, match="declares no overlapping"):
        _profile("partitioned-serial", allowed_overlap={("reward", "checkpoint")})
    p = _profile("partitioned-overlap", allowed_overlap={("reward", "checkpoint")})
    assert overlap_violation(p, "reward", "checkpoint") is None
    # Checkpoint of round r may overlap generation of round r+1 (different roles).
    assert overlap_violation(p, "generate", "checkpoint") is None
    # eval and generate both need the rollout role.
    assert "rollout" in overlap_violation(p, "eval", "generate")


def test_decoupled_outer_protocol_is_not_island_async():
    p = _profile("partitioned-serial", outer="decoupled")
    assert p.max_policy_age == 0
    assert generate_blockers(p, _snap(trained_policy_version=4)) != []
    with pytest.raises(ProfileError, match="does not relax island staleness"):
        _profile("partitioned-overlap", outer="decoupled", max_policy_age=1)
    with pytest.raises(ProfileError, match="not certified"):
        _profile("partitioned-overlap", algorithm_contract="one-step-off-policy")


def test_train_needs_complete_same_policy_groups_and_backpressure_is_bounded():
    p = _profile("partitioned-serial")
    groups = tuple(f"g{i}" for i in range(4))
    ok = _snap(ready_group_ids=groups, group_policy_versions={g: 3 for g in groups})
    assert train_blockers(p, ok) == []
    mixed = _snap(ready_group_ids=groups,
                  group_policy_versions={"g0": 2, "g1": 3, "g2": 3, "g3": 3})
    assert any("mixes" in r for r in train_blockers(p, mixed))
    partial = _snap(ready_group_ids=groups[:3], group_policy_versions={g: 3 for g in groups})
    assert any("3/4" in r for r in train_blockers(p, partial))
    full = _snap(ready_group_ids=groups)
    assert any("ready buffer full" in r for r in generate_blockers(p, full))
    assert any("backpressure" in r for r in generate_blockers(p, _snap(inflight_batches=1)))


def test_quiescent_cut_lists_every_open_obligation():
    p = _profile()
    assert quiescent_cut_blockers(p, _snap(ready_group_ids=("g0",))) == []
    reasons = quiescent_cut_blockers(
        p, _snap(active_requests=0, tool_wait=1, grad_accumulation_open=True, sync_phase="submitting")
    )
    assert any("tools" in r for r in reasons)
    assert any("accumulation" in r for r in reasons)
    assert any("submitting" in r for r in reasons)


def test_profile_from_manifest_is_stable_and_rejects_floor_division():
    m = example_manifest()
    p = ExecutionProfile.from_manifest_profile(m["profile"], m["work"])
    q = ExecutionProfile.from_manifest_profile(m["profile"], m["work"])
    assert p.contract_hash == q.contract_hash and p.contract_hash.startswith("sha256:")
    assert p.execution_mode == "partitioned-serial" and p.batch_samples == 48
    with pytest.raises(ProfileError, match="floor division"):
        _profile(groups_per_batch=3, samples_per_group=3, optimizer_steps_per_round=2)


# -- alignment A1: AlgorithmSpec binding ------------------------------------

from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: E402
from yeto.rl.engine.execution_profile import (  # noqa: E402
    algorithm_max_policy_staleness,
    check_algorithm_contract,
    execution_max_policy_staleness,
)


class _StaleSpec:
    """Stand-in for a future spec whose execution group tolerates staleness 0."""

    def __init__(self, staleness, digest="a" * 64):
        self.execution = type("E", (), {"max_policy_staleness": staleness})()
        self._digest = digest

    def sha256(self):
        return self._digest


def test_profile_binds_the_algorithm_spec_hash_and_hash_changes_identity():
    spec = AlgorithmSpec()
    unbound = _profile()
    bound = unbound.bind_algorithm(spec)
    assert bound.algorithm_spec_sha256 == spec.sha256()
    assert bound.contract_hash != unbound.contract_hash
    check_algorithm_contract(bound, spec)
    other = AlgorithmSpec(kl_coef=0.1)
    with pytest.raises(ProfileError, match="bound to algorithm"):
        check_algorithm_contract(bound, other)
    with pytest.raises(ProfileError, match="not bound"):
        check_algorithm_contract(unbound, spec)
    with pytest.raises(ProfileError, match="64 lowercase hex"):
        _profile(algorithm_spec_sha256="xyz")


def test_policy_age_never_exceeds_algorithm_staleness_and_modes_produce_zero():
    assert algorithm_max_policy_staleness(AlgorithmSpec()) == 0  # v1: on-policy
    assert algorithm_max_policy_staleness(_StaleSpec(0)) == 0
    with pytest.raises(ProfileError):
        algorithm_max_policy_staleness(_StaleSpec(-1))
    assert execution_max_policy_staleness(
        ["colocated-serial", "partitioned-serial", "partitioned-overlap"]
    ) == 0
    with pytest.raises(ProfileError):
        execution_max_policy_staleness(["one-step-off"])
    spec = _StaleSpec(0)
    check_algorithm_contract(_profile(algorithm_spec_sha256=spec.sha256()), spec)


def test_manifest_profile_carries_algorithm_hash():
    manifest = example_manifest()
    manifest["profile"]["algorithm_spec_sha256"] = AlgorithmSpec().sha256()
    profile = ExecutionProfile.from_manifest_profile(manifest["profile"], manifest["work"])
    check_algorithm_contract(profile, AlgorithmSpec())
