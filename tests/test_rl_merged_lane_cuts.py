"""A merged-lane synced trainer names its buffered groups as merged release cuts.

Its samplers serve merged releases (versions advance once per merged version,
e.g. every 8 local rounds with 8 fragments), while the trainer's own publications
advance every round. Labelling the groups with the trainer's local publications
named v0 with the initial weights only by coincidence and failed once the local
window passed it (Modal: "policy v0 of a trained trajectory was not published").
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from test_rl_engine_driver import _driver, _engine, _strict_config  # noqa: F401
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.sampler_island import wire_policy_releases
from yeto.rl.engine.external_policy import CUT_METADATA_KEY, ExternalMergedCuts, behavior_cut_of
from yeto.rl.engine.merged_release import OWN_LANE, BufferLane
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_roles import SampleBufferConfig
from yeto.rl.quant.schemes import BF16, release_dir_for

H0, H1 = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def _clean_executor_state():
    saved = (dict(br._MERGED_RELEASES), dict(br._MERGED_RELEASE_STORE), br._ACTIVE.pop(br.TRAINER, None),
             dict(br._PENDING_RELAY_GATE))
    br._MERGED_RELEASES.clear(); br._MERGED_RELEASE_STORE.clear(); br._PENDING_RELAY_GATE.clear()
    br._MERGED_MODE.clear()
    yield
    br._MERGED_MODE.clear()
    br._MERGED_RELEASES.clear(); br._MERGED_RELEASES.update(saved[0])
    br._MERGED_RELEASE_STORE.clear(); br._MERGED_RELEASE_STORE.update(saved[1])
    br._PENDING_RELAY_GATE.clear(); br._PENDING_RELAY_GATE.update(saved[3])
    if saved[2] is not None:
        br._ACTIVE[br.TRAINER] = saved[2]


def _sample(version: int) -> dict:
    return {"weight_versions": [{"spans": [{"version": str(version), "abs_start": 0, "abs_end": 2}],
                                 "prefill_spans": [], "output_start": 0, "prompt_tokens": 1}]}


def test_groups_are_stamped_with_their_merged_release_and_the_gate_waits_for_the_reader():
    assert not br.merged_release_mode()
    br.executor_set_merged_releases(None, {0: H0, 1: H1}, 1)
    assert br.merged_release_mode()
    assert br._PENDING_RELAY_GATE[br.TRAINER] == 1  # applied when the reader registers
    group = [_sample(0), _sample(1)]
    br.stamp_merged_release_cuts(group)
    assert [behavior_cut_of([s]) for s in group] == [(0, H0), (1, H1)]
    assert behavior_cut_of(group) == (0, H0)  # a trajectory's oldest cut


def test_an_unknown_release_is_read_from_the_store_or_refused(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    releases = PolicyReleaseStore(store, release_dir=release_dir_for(BF16))
    releases.release(3, "c" * 64, lambda d: (d / "model.bin").write_bytes(b"w"))
    br.executor_set_merged_releases(None, {0: H0}, 0, str(tmp_path / "store"), release_dir_for(BF16))
    group = [_sample(3)]
    br.stamp_merged_release_cuts(group)  # released after the driver's last push
    assert group[0]["metadata"][CUT_METADATA_KEY]["digest"] == "c" * 64
    with pytest.raises(br.SampleBufferRolloutError, match="merged release v7"):
        br.stamp_merged_release_cuts([_sample(7)])


def test_a_release_pruned_before_a_relaunched_trainer_saw_it_is_passed_over():
    # a relaunched merged-lane trainer knows v0 and the kept releases 5, 6; its
    # lane backlog still holds a v4 group (inside the age gate, record pruned)
    br.executor_set_merged_releases(None, {0: H0, 5: H1, 6: "c" * 64}, 6)
    group = [_sample(4), _sample(5)]
    assert br.stamp_merged_release_cuts(group) is False
    assert all("metadata" not in s for s in group)  # nothing half-stamped
    assert br.stamp_merged_release_cuts([_sample(5)]) is True
    with pytest.raises(br.SampleBufferRolloutError, match="merged release v7"):
        br.stamp_merged_release_cuts([_sample(7)])  # newer: not visible yet


class _Rollout:
    def __init__(self, pool):
        self.pool, self.calls = pool, []

    def _in_executor(self, fn, *args):
        self.calls.append(fn.__name__)
        return fn(None, *args)

    def __getattr__(self, name):
        return getattr(self.pool, name)


@pytest.mark.parametrize("lane,merged", [(BufferLane(1, 2), True), (OWN_LANE, False)])
def test_only_a_merged_lane_counts_its_groups_in_merged_cuts(tmp_path, lane, merged):
    engine = _engine()
    driver = _driver(engine, None, tmp_path, name="events.jsonl", learner_id=lane.index)
    driver.rollout = _Rollout(driver.rollout)
    wire_policy_releases(
        driver, actor_model=SimpleNamespace(), runner=SimpleNamespace(run=asyncio.run),
        miles_args=SimpleNamespace(
            yeto_rl_sample_buffer=SampleBufferConfig(role="trainer", store=str(tmp_path / "s")),
            yeto_rl_sample_lane=lane, lora_rank=0))
    assert isinstance(driver.external_policy, ExternalMergedCuts) is merged


def test_merged_mode_starts_before_any_release_and_v0_is_the_initial_weights(tmp_path):
    """A non-publisher trainer may publish before the publisher's first release
    exists (on Modal the publisher's initial cut upload took 12 min): the mode is
    on from the first push, and merged v0 is the island's own v0 (the base)."""
    engine = _engine()
    driver = _driver(engine, None, tmp_path, name="events.jsonl", learner_id=1)
    driver.rollout = _Rollout(driver.rollout)
    wire_policy_releases(
        driver, actor_model=SimpleNamespace(), runner=SimpleNamespace(run=asyncio.run),
        miles_args=SimpleNamespace(
            yeto_rl_sample_buffer=SampleBufferConfig(role="trainer", store=str(tmp_path / "s")),
            yeto_rl_sample_lane=BufferLane(1, 2), lora_rank=0))
    # the wrapper pushes right after the publish, before the driver records v0
    state = SimpleNamespace(policy_version=0, policy_tensor_hash=lambda: H0)
    driver.publisher.set_gate_version(0, state)
    assert driver.publisher.with_state
    assert br.merged_release_mode() and br._MERGED_RELEASES == {0: H0}
    group = [_sample(0)]
    br.stamp_merged_release_cuts(group)
    assert behavior_cut_of(group) == (0, H0)


def test_a_synced_trainer_counts_evidence_age_in_merged_versions_not_local_rounds():
    """SC2 P2f/P2g: a synced trainer on a merged lane trains v0 groups at local
    round 5 while no outer step has landed (merged version 0). Its evidence
    (schema 5, external merged cuts) bounds the age by the merged version, so
    the group fits; an unsynced island's schema-4 evidence still bounds it by the
    local round and refuses it (unchanged)."""
    from yeto.rl import trajectory_evidence as te
    from yeto.rl.engine.external_policy import EXTERNAL_MERGED_CUT

    h = "a" * 64
    # merged lane: local round 5, merged version 0, a v0 cut under max_policy_age 4
    assert te._external_cut_problem(EXTERNAL_MERGED_CUT, 0, 0, 4) is None
    assert te._external_cut_problem(EXTERNAL_MERGED_CUT, 0, 5, 4) is not None  # merged moved on
    # unsynced (schema 4): the same v0 trajectory at local round 5 is over age
    assert te._behavior_policy_problem(0, h, 5, h, 4) == \
        "trajectory policy is older than the batch's policy-age limit"
    assert te._behavior_policy_problem(1, h, 5, h, 4) is None
