"""S27 R-1: policy versions of an island under full-parameter asynchronous rounds.

No Ray. Covered here:

* the merged version of the fragments (one per outer round of all fragments);
* catch-up after a re-JOIN: a fragment jumps over rounds the island never
  received, only when no older version can still arrive;
* the session's round event and ``driver.outer_version``: what an island
  overwrote since its previous boundary, and that the policy it publishes at
  version r + 1 is the overwritten one (the behaviour policy of its samples);
* ``--rl-max-policy-age`` on the asynchronous rounds and full-parameter
  synced trainers (launcher and island checks, merged-version releases);
* the real Rust syncer with a membership lease: it starts at quorum and
  finalizes without an island that never came.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_rl_full_parameter_async import (
    P,
    _async_pairing,
    _client,
    _f32,
    _initial,
    _island_args,
    _start_async_syncer,
    _wait,
    needs_cargo,
    syncer_binary,  # noqa: F401 - pytest fixture
)
from test_rl_full_parameter_sync import _candidate, _Driver, _PairedRole
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
    fragment_rounds,
    merged_version,
)


class _MonotonicRole(_PairedRole):
    """A role whose stream delivers each fragment's versions in order (the
    real syncer client drops a version older than one it delivered)."""

    monotonic_broadcasts = True

    async def apply_staged_broadcast(self, fragment_id, version, *, catch_up=False):
        base = max(0, version - len(self.fragment_versions))
        held = self.fragment_versions[fragment_id]
        assert (held < base) if catch_up else (held == base)
        return await super().apply_staged_broadcast(fragment_id, version)


# -- merged version ------------------------------------------------------------------


def test_merged_version_counts_rounds_every_fragment_completed():
    assert [fragment_rounds(0, v, 2) for v in (0, 1, 3, 5)] == [0, 1, 2, 3]
    assert [fragment_rounds(1, v, 2) for v in (0, 2, 4)] == [0, 1, 2]
    assert merged_version((0, 0)) == 0
    assert merged_version((1, 0)) == 0  # fragment 1 not merged yet
    assert merged_version((1, 2)) == 1
    assert merged_version((3, 2)) == 1
    assert merged_version((3, 4)) == 2
    assert merged_version((7, 4, 3)) == 1  # P = 3: fragment 2 has one merge
    assert merged_version(()) == 0


# -- catch-up after a re-JOIN ---------------------------------------------------------


@pytest.mark.asyncio
async def test_rejoin_catch_up_jumps_a_fragment_over_missed_rounds():
    actor = _MonotonicRole("actor", [], [])
    pairing = _async_pairing(actor)
    actor._staged_steps.extend([5, 4])  # the syncer's newest values after a re-JOIN

    committed = await pairing.commit_staged_broadcasts()

    assert actor.fragment_versions == (5, 4)
    assert set(committed["actor"].applied_fragments) == {0, 1}
    assert pairing.overwritten_rounds == 2
    assert pairing.caught_up_rounds == 2 + 1  # fragment 0 skipped 1, 3; fragment 1 skipped 2


@pytest.mark.asyncio
async def test_catch_up_waits_without_a_monotonic_stream():
    actor = _PairedRole("actor", [], [])
    actor.monotonic_broadcasts = False  # no ordering promise: version 1 may still arrive
    pairing = _async_pairing(actor)
    actor._staged_steps.append(5)

    await pairing.commit_staged_broadcasts()

    assert actor.fragment_versions == (0, 0) and pairing.caught_up_rounds == 0


@pytest.mark.asyncio
async def test_catch_up_applies_an_older_staged_version_in_order_first():
    actor = _MonotonicRole("actor", [], [])
    pairing = _async_pairing(actor)
    actor._staged_steps.extend([3, 1])

    await pairing.commit_staged_broadcasts()

    assert actor.fragment_versions == (3, 0)
    assert pairing.caught_up_rounds == 0 and pairing.overwritten_rounds == 2


@pytest.mark.asyncio
async def test_catch_up_releases_retained_submissions_it_supersedes():
    actor = _MonotonicRole("actor", [_candidate("actor", 1)], [], release_broadcasts=False)
    pairing = _async_pairing(actor)
    pairing.submit_ready()
    assert len(pairing.inflight) == 1
    actor._queued_steps.clear()
    actor._staged_steps.append(5)  # round 1 and 3 closed while this island was away

    await pairing.commit_staged_broadcasts()

    assert pairing.inflight == {} and pairing.journal == []
    assert actor.fragment_versions == (5, 0)


@pytest.mark.asyncio
async def test_catch_up_needs_every_role_to_hold_the_same_version():
    actor, critic = _MonotonicRole("actor", [], []), _MonotonicRole("critic", [], [])
    critic.fragment_versions = (1, 0)
    pairing = _async_pairing(actor, critic)
    actor._staged_steps.append(5)
    critic._staged_steps.append(5)

    await pairing.commit_staged_broadcasts()

    assert actor.fragment_versions == (0, 0) and critic.fragment_versions == (1, 0)


# -- session: round event, outer version, published behaviour policy ------------------------


def _session(*streams, partial_quorum=True):
    pairing = _async_pairing(*streams, partial_quorum=partial_quorum)
    return FullParameterSync(pairing, groups={s.role: "g" for s in streams}, run=asyncio.run,
                             trajectories=lambda rollout_id: rollout_id)


def test_published_policy_includes_the_fragments_overwritten_at_the_boundary():
    actor = _MonotonicRole("actor", [], [])
    session, driver = _session(actor), _Driver()
    session.start(driver)
    assert driver.outer_version == 0

    actor._staged_steps.extend([1, 2])  # rounds 1 and 2 closed without this island
    boundary = session.boundary(driver, rollout_id=0, stats=None)

    # Version 1 is published from the overwritten cut: samples generated from
    # it carry its hash, so their behaviour log-probabilities are those weights.
    assert boundary.state.policy_version == 1
    assert boundary.state.policy_tensor_hash() == "cut-(1, 2)"
    fields = driver.events[-1][1]
    assert fields["sync/overwritten_fragments"] == 2
    assert fields["sync/outer_version"] == 1 == driver.outer_version == session.outer_version

    session.boundary(driver, rollout_id=1, stats=None)  # nothing new arrived
    fields = driver.events[-1][1]
    assert fields["sync/overwritten_fragments"] == 0 and fields["sync/overwritten_rounds"] == 2
    assert fields["sync/caught_up_rounds"] == 0


def test_strict_sweep_has_no_merged_outer_version():
    actor = _PairedRole("actor", [], [])
    pairing = RolePairing({"actor": actor}, FullParameterSyncProfile(
        roles=("actor",), total_fragment_steps=4, fragments=2, local_horizon=1,
        pipeline_depth=2, wait_timeout=1.0, poll_seconds=0.001, sweep=True))
    session = FullParameterSync(pairing, groups={"actor": "g"}, run=asyncio.run,
                                trajectories=lambda rollout_id: rollout_id)
    driver = _Driver()
    session.start(driver)
    assert session.outer_version is None and not hasattr(driver, "outer_version")


def test_synced_trainer_releases_the_merged_version_of_the_fragments(tmp_path):
    from yeto.rl.adapters.miles.sampler_island import GateVersionPublisher, ReleasingPublisher
    from yeto.rl.engine.merged_release import MergedReleaseGuard, merged_version as release_version
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.policy_release import PolicyReleaseStore

    class _Inner:
        def publish(self, state, **_):
            return "ok"

    actor = _MonotonicRole("actor", [], [])
    session, driver = _session(actor), _Driver()
    events, gates = [], []
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    version_of = lambda state: release_version(driver, state)  # noqa: E731
    publisher = ReleasingPublisher(
        GateVersionPublisher(_Inner(), gates.append, version_of), releases,
        lambda v, d: (d / "model.safetensors").write_bytes(b"%d" % v), lambda: 0,
        lambda event, **f: events.append((event, f)), version_of=version_of,
        guard=MergedReleaseGuard(lambda a, b: events.append(("gap", {}))))
    publisher.publish(session.start(driver).state)
    for rollout_id, staged in enumerate(([1], [2], [3, 4])):
        actor._staged_steps.extend(staged)
        publisher.publish(session.boundary(driver, rollout_id=rollout_id, stats=None).state)

    assert gates == [0, 0, 1, 2]
    released = [f["policy_version"] for e, f in events if e == "rl_policy_released"]
    assert released == [0, 1, 2]
    assert [f["local_version"] for e, f in events if e == "rl_policy_released"] == [0, 2, 3]


# -- policy age on the asynchronous rounds --------------------------------------------


def test_asynchronous_rounds_take_a_policy_age_limit_and_the_sweep_does_not():
    from yeto.rl.adapters.miles.island_entry import _check_dense_full_sweep

    _check_dense_full_sweep(_island_args(rl_max_policy_age=2), yeto_policy_sync=True)
    with pytest.raises(ValueError, match="--rl-max-policy-age 0"):
        _check_dense_full_sweep(_island_args(full_parameter_rounds="sweep", local_horizon=1,
                                             rl_max_policy_age=1), yeto_policy_sync=True)


def _synced_args(**kw):
    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=False,
                rl_checkpoint_store="modal-volume://v/p", rl_max_policy_age=2, tuning="lora",
                rl_sync_preset="full-parameter", rl_sync_full_parameter_rounds="async")
    return SimpleNamespace(**{**base, **kw})


def test_full_parameter_synced_trainers_need_the_asynchronous_rounds():
    from yeto.rl.engine import sample_roles

    sample_roles.check_synced_trainers(_synced_args(), 2)
    with pytest.raises(ValueError, match="rounds async"):
        sample_roles.check_synced_trainers(_synced_args(rl_sync_full_parameter_rounds="sweep"), 2)


def test_full_parameter_synced_trainer_plan(monkeypatch):
    from test_rl_sampler_trainer_pair_launch import _args

    from yeto.launcher import dry_run_plan

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "r1-test-key")
    synced = ("--rl-sample-role", "trainer", "--rl-max-policy-age", "2",
              "--rl-checkpoint-store", "modal-volume://yeto-r1/run-a", "--rl-sample-wait-s", "900",
              "--cluster-prefix", "r1-synced",
              "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json")
    # store-fed trainers start no engine: no fixed partition (S27 G-BG)
    full = ("--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
            "--fragments", "4")
    gpu = "modal:2xh100,modal:2xh100"
    plan = dry_run_plan(_args(full + ("--rl-sync-full-parameter-rounds", "async"), gpu=gpu,
                              base=synced))
    assert plan["syncer"] is not None and len(plan["island_requests"]) == 2
    # Unset rounds default to async in a multi-island run; the strict sweep refuses.
    assert dry_run_plan(_args(full, gpu=gpu, base=synced))["syncer"] is not None
    with pytest.raises(ValueError, match="rounds async"):
        _args(full + ("--rl-sync-mode", "strict"), gpu=gpu, base=synced)
    with pytest.raises(ValueError, match="trainers start no engine"):
        _args(full + ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"), gpu=gpu,
              base=synced)


# -- the real Rust syncer with a membership lease -------------------------------------


@needs_cargo
def test_leased_syncer_starts_at_quorum_without_an_absent_island(
        syncer_binary, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "r1-test-key")
    process, port, profile_hash = _start_async_syncer(
        syncer_binary, tmp_path, "--rl-sync-quorum", "1", total_steps=2,
        syncer_args=("--island-lease-s", "3"))
    client = _client(0, port, profile_hash)  # learner 1 never connects
    seen: set = set()

    def poll():
        client.check_health()
        client.heartbeat(0)
        client.drain_updates()
        for permit in client.drain_pulls():
            if (permit.global_step, permit.round_attempt) in seen:
                continue
            seen.add((permit.global_step, permit.round_attempt))
            base = max(0, permit.global_step - P)
            delta = np.full(3, 0.5, dtype=np.float32)
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 base, permit.global_step, 1, 7, _f32(delta))
        return client.finalizing.is_set()

    try:
        client.start()
        for fid in range(P):
            client.send_init(fid, _f32(_initial(fid)))
        started = time.monotonic()
        _wait(poll, "syncer never finalized with one of two islands", timeout=60)
        manifest, _ = client.wait_for_final_fragments(timeout=30)
        client.acknowledge_finalization(manifest)
        assert process.wait(timeout=20) == 0
        assert time.monotonic() - started < 40
    finally:
        client.close()
        if process.poll() is None:
            process.terminate()
        log, _ = process.communicate(timeout=10)
    # Learner 1 never connected: rounds started at quorum, the final cut
    # waited for the live member only (finalization-time expiry: Rust tests).
    assert "start_members=1" in log and "learners=1" in log, log


# -- the Miles role stream ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_quorum", [True, False])
async def test_role_stream_catch_up_jumps_only_with_a_partial_quorum(partial_quorum):
    from test_rl_full_parameter_async import _stream
    from yeto.protocol import PullRequest

    stream = _stream(partial_quorum)
    assert stream.monotonic_broadcasts
    stream._fragment_versions = [0, 0]
    stream._pending = {0: PullRequest(0, 3, 1)}
    stream._staged = {(0, 5): object()}

    async def apply(selected, **_):
        stream._fragment_versions[0] = 5

    stream._apply_authoritative = apply
    with pytest.raises(RuntimeError, match="skipped its base version"):
        await stream.apply_staged_broadcast(0, 5)
    if not partial_quorum:
        with pytest.raises(RuntimeError, match="catch-up needs a partial quorum"):
            await stream.apply_staged_broadcast(0, 5, catch_up=True)
        return
    await stream.apply_staged_broadcast(0, 5, catch_up=True)
    assert stream._fragment_versions == [5, 0] and stream._pending == {} and stream._staged == {}
