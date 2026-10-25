"""S27 G-EM: relay-mode policy identity (external merged cuts).

A relay trainer trains on groups that spot samplers generated with merged cuts
of the syncer, not with the island's own publications. The driver counts their
age in merged versions and the trajectory evidence (schema 5) names each
trajectory's cut. Off relay mode nothing changes (the last tests here and the
existing driver and evidence tests).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.test_rl_policy_age import _driver
from tests.test_rl_trajectory_evidence import _reward_function_sample
from yeto.rl import trajectory_evidence
from yeto.rl.contracts import EXTERNAL_MERGED_CUT
from yeto.rl.engine.driver import PolicyIdentityError
from yeto.rl.engine.external_policy import (
    CUT_METADATA_KEY,
    ExternalCutError,
    ExternalMergedCuts,
    behavior_cut_of,
    cut_digest,
    cut_record,
    fragment_digest,
)
from yeto.rl.trajectory_evidence import (
    bind_round_policy,
    build_trajectory_batch_evidence,
    read_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)

DIGEST = "d" * 64
REWARD = "e" * 64
ROUND = "c" * 64


class HalfMerged(ExternalMergedCuts):
    """Test double: the trainer's merged version is half its local version (the
    async 0.6B shape: about two local rounds per merged version)."""

    def published(self, local_version, merged_version):
        super().published(local_version, local_version // 2)


class TripleMerged(ExternalMergedCuts):
    def published(self, local_version, merged_version):
        super().published(local_version, 3 * local_version)


def _relay_groups(driver, merged_of):
    original = driver.rollout.generate

    def gen(rid, **kw):
        h = original(rid, **kw)
        groups = tuple(dataclasses.replace(g, policy_token=str(merged_of(rid))) for g in h.groups)
        return dataclasses.replace(h, groups=groups)

    driver.rollout.generate = gen


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]


def test_relay_driver_accepts_groups_named_by_merged_versions(tmp_path):
    driver = _driver(tmp_path, 2)
    driver.external_policy = HalfMerged()
    _relay_groups(driver, lambda rid: max(0, rid // 2 - 1))  # one merged version behind
    driver.run()
    events = _events(tmp_path)
    assert len([e for e in events if e["event"] == "rl_round_trained"]) == 3
    policies = [e for e in events if e["event"] == "rl_group_policies"]
    assert [e["rollout_id"] for e in policies] == [0, 1, 2]
    assert {e["kind"] for e in policies} == {EXTERNAL_MERGED_CUT}
    assert [e["merged_version"] for e in policies] == [0, 0, 1]
    assert [sorted(e["ages"]) for e in policies] == [["0"], ["0"], ["1"]]
    assert all(g[1] in ("0", "1") for e in policies for g in e["groups"])


def test_the_same_groups_fail_the_local_identity_check(tmp_path):
    """The failure G-EE predicted (SPOT-ROLLOUT.md 11.1 item 3): without relay
    mode a merged version is read as a local version."""
    driver = _driver(tmp_path, 1)
    _relay_groups(driver, lambda rid: 7)
    with pytest.raises(PolicyIdentityError):
        driver.run()


def test_relay_driver_refuses_a_group_older_than_the_limit_in_merged_versions(tmp_path):
    driver = _driver(tmp_path, 1)
    driver.external_policy = TripleMerged()
    _relay_groups(driver, lambda rid: 0)  # round 1: merged version 3, the groups' 0 is 3 behind
    with pytest.raises(PolicyIdentityError, match="merged version 0 is 3 behind 3"):
        driver.run()


def test_relay_driver_accepts_a_cut_newer_than_its_merged_version(tmp_path):
    driver = _driver(tmp_path, 1)
    driver.external_policy = HalfMerged()
    _relay_groups(driver, lambda rid: rid // 2 + 1)
    driver.run()
    ages = [sorted(e["ages"]) for e in _events(tmp_path) if e["event"] == "rl_group_policies"]
    assert ages == [["-1"]] * 3


def test_check_groups_names_every_problem():
    groups = [SimpleNamespace(group_id="a", policy_token="5", policy_versions=None),
              SimpleNamespace(group_id="b", policy_token="2", policy_versions=None),
              SimpleNamespace(group_id="c", policy_token="mixed:1|2", policy_versions=None),
              SimpleNamespace(group_id="d", policy_token="4", policy_versions=(3, 4))]
    problems, ages = ExternalMergedCuts.check_groups(groups, 5, 2)
    assert ages == {0: 1, 2: 1}
    assert len(problems) == 2 and "b: merged version 2 is 3 behind 5" in problems[0]
    assert "no merged version" in problems[1]
    cuts = ExternalMergedCuts(keep=2)
    for local in range(4):
        cuts.published(local, local + 10)
    assert cuts.merged_at(3) == 13 and cuts.merged_at(1) is None and cuts.merged_at(None) is None


def test_cut_digest_binds_layout_versions_and_bytes():
    one, two = fragment_digest(b"\x01\x02"), fragment_digest(b"\x03\x04")
    base = cut_digest("ab" * 32, (3, 4), (one, two))
    assert len(base) == 64
    assert cut_digest("ab" * 32, (3, 5), (one, two)) != base
    assert cut_digest("ab" * 32, (3, 4), (two, one)) != base
    assert cut_digest("cd" * 32, (3, 4), (one, two)) != base
    with pytest.raises(ExternalCutError):
        cut_digest("ab" * 32, (3,), (one, two))


def _cut_sample(*, group_index, index, merged, digest=DIGEST):
    sample = _reward_function_sample(group_index=group_index, index=index, value=1.0)
    sample.metadata[CUT_METADATA_KEY] = cut_record(merged, digest)
    return sample


def _external(merged_versions, *, merged_version=5, limit=2, rollout_id=9):
    groups = [[_cut_sample(group_index=0, index=i, merged=m) for i, m in enumerate(merged_versions)]]
    return build_trajectory_batch_evidence(
        groups, rollout_id=rollout_id, behavior_policy_hash=ROUND, reward_contract_hash=REWARD,
        evidence_kind="reward-function", behavior_cut_of=behavior_cut_of,
        merged_version=merged_version, max_policy_age=limit)


def test_schema_5_names_each_trajectory_cut_and_round_trips(tmp_path):
    evidence = _external([5, 4, 6])
    assert evidence.schema_version == 5 and evidence.merged_version == 5
    assert evidence.behavior_policy_hash == ROUND  # the weights the round trains from
    for item, merged in zip(evidence.envelopes, (5, 4, 6)):
        assert item.behavior_policy_kind == EXTERNAL_MERGED_CUT
        assert (item.behavior_merged_version, item.behavior_cut_digest) == (merged, DIGEST)
        assert item.behavior_policy_version is None and item.behavior_policy_hash is None
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    raw = json.loads(path.read_text())
    assert raw["schema"] == 5 and raw["merged_version"] == 5 and raw["max_policy_age"] == 2
    assert raw["envelopes"][0]["behavior_policy_kind"] == EXTERNAL_MERGED_CUT
    assert read_trajectory_batch_evidence(path) == evidence
    raw["envelopes"][1]["behavior_merged_version"] = 5
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_schema_5_trajectory_ids_depend_on_the_cut():
    a, b = _external([5]), _external([4])
    assert a.envelopes[0].trajectory_id != b.envelopes[0].trajectory_id


def test_schema_5_refuses_a_cut_older_than_the_limit_and_a_missing_cut():
    with pytest.raises(ValueError, match="older than the batch's policy-age limit"):
        _external([2])
    sample = _reward_function_sample(group_index=0, index=0, value=1.0)
    with pytest.raises(ExternalCutError, match="names no external merged cut"):
        build_trajectory_batch_evidence(
            [[sample]], rollout_id=1, behavior_policy_hash=ROUND, reward_contract_hash=REWARD,
            evidence_kind="reward-function", behavior_cut_of=behavior_cut_of,
            merged_version=1, max_policy_age=1)
    with pytest.raises(ValueError, match="either published or an external merged cut"):
        build_trajectory_batch_evidence(
            [[_cut_sample(group_index=0, index=0, merged=1)]], rollout_id=1, behavior_policy_hash=ROUND,
            reward_contract_hash=REWARD, evidence_kind="reward-function",
            behavior_cut_of=behavior_cut_of, behavior_policy_of=lambda s: (1, ROUND),
            merged_version=1, max_policy_age=1)


def test_schema_5_binds_a_later_round_policy():
    evidence = _external([5])
    bound = bind_round_policy(evidence, "f" * 64)
    assert bound.behavior_policy_hash == "f" * 64 and bound.envelopes == evidence.envelopes


def test_behavior_cut_of_takes_the_oldest_segment_cut():
    segs = [{"metadata": {CUT_METADATA_KEY: cut_record(4, "a" * 64)}},
            {"metadata": {CUT_METADATA_KEY: cut_record(3, "b" * 64)}}]
    assert behavior_cut_of(segs) == (3, "b" * 64)
    with pytest.raises(ExternalCutError, match="different cuts"):
        behavior_cut_of([{"metadata": {CUT_METADATA_KEY: cut_record(3, "a" * 64)}}] + segs[1:])
    with pytest.raises(ExternalCutError):
        behavior_cut_of([{"metadata": {CUT_METADATA_KEY: {**cut_record(3, "a" * 64), "x": 1}}}])


def test_envelope_contract_keeps_local_and_external_policies_apart():
    item = _external([5]).envelopes[0]
    with pytest.raises(ValueError, match="does not name a local publication"):
        dataclasses.replace(item, behavior_policy_version=5)
    with pytest.raises(ValueError, match="kind is unsupported"):
        dataclasses.replace(item, behavior_policy_kind="other")
    with pytest.raises(ValueError):
        dataclasses.replace(item, behavior_policy_kind=None, behavior_merged_version=None,
                            behavior_cut_digest=None)  # then a local version and hash are required


def test_local_schemas_keep_their_exact_envelope_fields(tmp_path):
    """Off relay mode the persisted envelope has no external-cut field (the
    bytes and batch hashes of schemas 2-4 are unchanged)."""
    sample = _reward_function_sample(group_index=0, index=0, value=1.0)
    evidence = build_trajectory_batch_evidence(
        [[sample]], rollout_id=3, behavior_policy_hash=ROUND, reward_contract_hash=REWARD,
        evidence_kind="reward-function", behavior_policy_of=lambda s: (2, "a" * 64), max_policy_age=1)
    assert evidence.schema_version == 4 and evidence.merged_version is None
    mapping = trajectory_evidence._envelope_mapping(evidence.envelopes[0], 4)
    assert set(mapping) == set(trajectory_evidence._LAYERED_ENVELOPE_FIELDS)
    assert not set(mapping) & trajectory_evidence._EXTERNAL_CUT_FIELDS
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    raw = json.loads(path.read_text())
    assert "merged_version" not in raw and "behavior_policy_kind" not in raw["envelopes"][0]
    assert set(asdict(evidence.envelopes[0])) - trajectory_evidence._EXTERNAL_CUT_FIELDS == set(mapping)


def test_miles_evidence_port_writes_schema_5_for_a_relay_round(tmp_path):
    from yeto.rl.adapters.miles.buffer_rollout import EXTERNAL_CUT_ROUND_ATTR
    from yeto.rl.adapters.miles.full_parameter.evidence import persist_published_round_evidence
    from yeto.rl.engine.policy_versions import PublishedPolicies

    args = SimpleNamespace(yeto_rl_trajectory_evidence_dir=str(tmp_path / "ev"),
                           yeto_rl_trajectory_evidence_kind="reward-function",
                           yeto_rl_trajectory_evidence_schema_version=2,
                           yeto_rl_reward_sha256=REWARD, yeto_rl_max_policy_age=2)
    setattr(args, EXTERNAL_CUT_ROUND_ATTR, {"rollout_id": 4, "merged_version": 3})
    groups = [[_cut_sample(group_index=0, index=0, merged=3), _cut_sample(group_index=0, index=1, merged=2)]]
    persist_published_round_evidence(args, groups, f"yeto:4:{ROUND}", PublishedPolicies(), rollout_id=4)
    evidence = read_trajectory_batch_evidence(tmp_path / "ev" / "rollout-00000004.json")
    assert evidence.schema_version == 5 and evidence.merged_version == 3 and evidence.max_policy_age == 2
    assert [e.behavior_merged_version for e in evidence.envelopes] == [3, 2]
    setattr(args, EXTERNAL_CUT_ROUND_ATTR, {"rollout_id": 4, "merged_version": 3})
    with pytest.raises(RuntimeError, match="relay take was for round 4"):
        persist_published_round_evidence(args, groups, f"yeto:5:{ROUND}", PublishedPolicies(), rollout_id=5)


def test_relay_gate_set_before_the_reader_exists_reaches_the_first_take():
    from yeto.rl.adapters.miles import buffer_rollout as br

    br._ACTIVE.pop(br.TRAINER, None)
    br._PENDING_RELAY_GATE.clear()
    assert br.executor_set_relay_gate_version(None, 6) is None
    reader = SimpleNamespace(gate_version=None)
    br.register_reader(reader)
    try:
        assert reader.gate_version == 6
        assert br.executor_set_relay_gate_version(None, 7) == 6 and reader.gate_version == 7
    finally:
        br._ACTIVE.pop(br.TRAINER, None)
    other = SimpleNamespace(gate_version=None)
    br.register_reader(other)  # no pending gate: a synced trainer off relay mode is unchanged
    br._ACTIVE.pop(br.TRAINER, None)
    assert other.gate_version is None
