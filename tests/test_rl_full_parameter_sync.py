from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.protocol import PartialMessageGenerationLost, PullRequest
from yeto.rl.elastic_client import LeaveReason
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    RoleRoundResult,
    RoleSubmissionCandidate,
)
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
    require_lockstep,
)


def _profile(roles=("actor", "critic"), *, wait_timeout=900.0, sweep=False, pipeline_depth=2):
    return FullParameterSyncProfile(
        roles=roles,
        total_fragment_steps=4,
        fragments=2,
        local_horizon=1 if sweep else 2,
        pipeline_depth=pipeline_depth,
        wait_timeout=wait_timeout,
        poll_seconds=0.001,
        sweep=sweep,
    )


def _pairing(*streams, **profile):
    roles = tuple(stream.role for stream in streams)
    return RolePairing({stream.role: stream for stream in streams}, _profile(roles, **profile))


async def _local_rounds(pairing, rollout_id):
    return await pairing.gather_roles(
        lambda _role, stream: stream.complete_local_round(rollout_id=rollout_id, trajectories=object())
    )


def _steps(progress, role):
    return [value.global_step for value in progress[role].submissions]


class _PairedRole:
    def __init__(
        self,
        role: str,
        candidates: list[RoleSubmissionCandidate],
        events: list[tuple[str, int, bool]],
        *,
        false_pushes: int = 0,
        empty_polls: int = 0,
        release_broadcasts: bool = True,
    ) -> None:
        self.role = role
        self.candidates = candidates
        self.events = events
        self.false_pushes = false_pushes
        self.empty_polls = empty_polls
        self.release_broadcasts = release_broadcasts
        self.fragment_versions = (0, 0)
        self.finalizing = False
        self.terminal_submitted = False
        self._queued_steps = []
        self._staged_steps = []
        self.local_boundaries = 0
        self.closed = False
        self.final_acknowledged = False

    def retain_candidate(self, candidate):
        del candidate

    def release_candidate(self, candidate):
        del candidate

    def close(self):
        self.closed = True

    async def initialize(self, group):
        self.group = group

    def resident_state(self, policy_version):
        versions = self.fragment_versions
        return SimpleNamespace(
            policy_version=policy_version,
            policy_tensor_hash=lambda: f"cut-{versions}",
        )

    def discard_detached_submission_cut(self):
        pass

    def acknowledge_final(self):
        self.final_acknowledged = True

    async def complete_local_round(self, *, rollout_id, trajectories):
        del trajectories
        self.local_boundaries += 1
        self.events.append(("train-boundary", self.role, rollout_id))
        return RoleRoundResult(self.role, object(), (), (), False)

    def ready_submission_candidates(self):
        if self.empty_polls:
            self.empty_polls -= 1
            return ()
        return tuple(self.candidates)

    def push_candidate(self, candidate):
        index = next(
            index
            for index, value in enumerate(self.candidates)
            if value.logical_key == candidate.logical_key
        )
        accepted = self.false_pushes == 0
        self.events.append((self.role, candidate.permit.global_step, accepted))
        if self.false_pushes:
            self.false_pushes -= 1
            return None
        self.candidates.pop(index)
        self._queued_steps.append(candidate.permit.global_step)
        self.terminal_submitted = self.terminal_submitted or (
            candidate.permit.global_step == 4
        )
        return SimpleNamespace(
            role=self.role,
            fragment_id=candidate.permit.fragment_id,
            global_step=candidate.permit.global_step,
            round_attempt=candidate.permit.round_attempt,
            base_version=candidate.base_version,
            c_steps=candidate.c_steps,
            c_tokens=candidate.c_tokens,
            payload_bytes=candidate.payload_bytes,
            pull_to_push_seconds=0.0,
        )

    async def stage_available_broadcasts(self):
        if self.release_broadcasts:
            self._staged_steps.extend(self._queued_steps)
            self._queued_steps.clear()
        return tuple(
            ((step - 1) % len(self.fragment_versions), step)
            for step in self._staged_steps
        )

    @property
    def staged_fragment_versions(self):
        return tuple(
            ((step - 1) % len(self.fragment_versions), step)
            for step in self._staged_steps
        )

    def has_staged_broadcast(self, fragment_id, version):
        return (fragment_id, version) in self.staged_fragment_versions

    async def apply_staged_broadcast(self, fragment_id, version):
        versions = list(self.fragment_versions)
        assert (fragment_id, version) in self.staged_fragment_versions
        self._staged_steps.remove(version)
        versions[fragment_id] = version
        self.fragment_versions = tuple(versions)
        return (fragment_id,)

    async def wait_and_apply_final(self):
        versions = list(self.fragment_versions)
        for step in self._queued_steps + self._staged_steps:
            versions[(step - 1) % len(versions)] = step
        self._queued_steps.clear()
        self._staged_steps.clear()
        self.fragment_versions = tuple(versions)


class _LostAfterQueueRole(_PairedRole):
    """Model a True enqueue whose generation dies before server commit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay = None
        self._lost_once = False

    def ready_submission_candidates(self):
        if self._replay is not None:
            replay, self._replay = self._replay, None
            self.candidates.append(replay)
        return super().ready_submission_candidates()

    def push_candidate(self, candidate):
        submission = super().push_candidate(candidate)
        if not self._lost_once:
            self._lost_once = True
            self._queued_steps.clear()
            self._replay = candidate
        return submission


class _PartialThenIncrementedAttemptRole(_PairedRole):
    """Lose a partial PUSH, then replay its exact cut with a newer attempt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lost_once = False

    def push_candidate(self, candidate):
        assert candidate.logical_key == self.candidates[0].logical_key
        if not self._lost_once:
            self._lost_once = True
            self.events.append((self.role, candidate.permit.global_step, False))
            replay = replace(
                candidate,
                permit=replace(
                    candidate.permit,
                    round_attempt=candidate.permit.round_attempt + 1,
                ),
            )
            self.candidates[0] = replay
            raise PartialMessageGenerationLost(
                connection_generation=1,
                connection_epoch=1,
                queued_chunks=1,
                operation="PUSH_FRAGMENT",
            )
        return super().push_candidate(candidate)


def _candidate(role: str, global_step: int, *, base_version: int = 0):
    return RoleSubmissionCandidate(
        role=role,
        permit=PullRequest((global_step - 1) % 2, global_step, 1),
        base_version=base_version,
        local_step=2,
        c_steps=2,
        c_tokens=11,
        payload_bytes=8,
        submission_cut=object(),
        anchor=object(),
    )



@pytest.mark.asyncio
async def test_pairing_nonblockingly_retains_asymmetric_pull():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1)], events)
    critic = _PairedRole("critic", [_candidate("critic", 1)], events, empty_polls=2)
    pairing = _pairing(actor, critic)

    assert _steps(pairing.submit_ready(), "actor") == []
    assert events == []
    pairing.submit_ready()
    progress = pairing.submit_ready()

    assert _steps(progress, "actor") == _steps(progress, "critic") == [1]
    assert events == [("actor", 1, True), ("critic", 1, True)]


@pytest.mark.asyncio
async def test_pipeline_depth_two_queues_later_pair_without_wan_commit():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1), _candidate("actor", 2)], events,
                        release_broadcasts=False)
    critic = _PairedRole("critic", [_candidate("critic", 1), _candidate("critic", 2)], events,
                         release_broadcasts=False)
    pairing = _pairing(actor, critic)

    await _local_rounds(pairing, 0)
    progress = pairing.submit_ready()
    assert _steps(progress, "actor") == _steps(progress, "critic") == [1, 2]
    assert len(pairing.inflight) == 2
    committed = await pairing.commit_staged_broadcasts()
    assert all(not value.applied_fragments for value in committed.values())
    # A second local-train boundary completes while both earlier WAN pairs
    # remain uncommitted: this is the intended streaming overlap.
    await _local_rounds(pairing, 1)
    assert actor.local_boundaries == critic.local_boundaries == 2
    assert len(pairing.inflight) == 2


@pytest.mark.asyncio
async def test_pair_replays_a_true_enqueue_that_never_reaches_server_commit():
    events = []
    actor = _LostAfterQueueRole("actor", [_candidate("actor", 1)], events)
    critic = _PairedRole("critic", [_candidate("critic", 1)], events)
    pairing = _pairing(actor, critic)

    pairing.submit_ready()
    committed = await pairing.commit_staged_broadcasts()
    assert committed["actor"].applied_fragments == committed["critic"].applied_fragments == ()
    pairing.submit_ready()
    committed = await pairing.commit_staged_broadcasts()

    assert committed["actor"].applied_fragments == committed["critic"].applied_fragments == (0,)
    assert actor.fragment_versions == critic.fragment_versions == (1, 0)
    assert events == [("actor", 1, True), ("critic", 1, True), ("actor", 1, True)]
    assert len(pairing.journal) == 1


@pytest.mark.asyncio
async def test_partial_push_replays_exact_cut_with_incremented_round_attempt():
    events = []
    actor = _PartialThenIncrementedAttemptRole("actor", [_candidate("actor", 1)], events)
    critic = _PairedRole("critic", [_candidate("critic", 1)], events)
    pairing = _pairing(actor, critic)

    progress = pairing.submit_ready()
    assert progress["actor"].submissions == ()
    assert [value.round_attempt for value in progress["critic"].submissions] == [1]
    progress = pairing.submit_ready()

    assert [value.round_attempt for value in progress["actor"].submissions] == [2]
    assert progress["critic"].submissions == ()
    assert events == [("actor", 1, False), ("critic", 1, True), ("actor", 1, True)]


@pytest.mark.asyncio
async def test_one_sided_timeout_fail_stops_every_role():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1), _candidate("actor", 2)], events)
    critic = _PairedRole("critic", [_candidate("critic", 1), _candidate("critic", 2)], events,
                         false_pushes=10_000)
    pairing = _pairing(actor, critic, wait_timeout=0.01)

    pairing.submit_ready()
    await asyncio.sleep(0.02)
    with pytest.raises(TimeoutError, match="one-sided paired submission"):
        pairing.submit_ready()

    assert ("actor", 2, True) in events
    assert len(pairing.inflight) == 2
    assert actor.closed and critic.closed
    with pytest.raises(RuntimeError, match="fail-stopped"):
        pairing.submit_ready()


@pytest.mark.asyncio
async def test_one_role_failure_closes_every_role():
    events = []
    actor = _PairedRole("actor", [], events)
    critic = _PairedRole("critic", [], events)

    async def broken(*, rollout_id, trajectories):
        raise RuntimeError("critic train step failed")

    critic.complete_local_round = broken
    pairing = _pairing(actor, critic)
    with pytest.raises(RuntimeError, match="critic train step failed"):
        await _local_rounds(pairing, 0)
    assert actor.closed and critic.closed
    assert str(pairing.failure) == "critic train step failed"


@pytest.mark.asyncio
async def test_terminal_final_manifest_drains_all_retained_pipeline_pairs():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 3, base_version=1), _candidate("actor", 4, base_version=2)],
                        events, release_broadcasts=False)
    critic = _PairedRole("critic", [_candidate("critic", 3, base_version=1),
                                    _candidate("critic", 4, base_version=2)],
                         events, release_broadcasts=False)
    actor.fragment_versions = critic.fragment_versions = (1, 2)
    pairing = _pairing(actor, critic)

    pairing.submit_ready()
    assert pairing.terminal_seen
    assert len(pairing.inflight) == 2
    actor.finalizing = critic.finalizing = True
    await pairing.finish_without_more_training()

    assert actor.terminal_submitted and critic.terminal_submitted
    assert actor.fragment_versions == critic.fragment_versions == (3, 4)
    assert not pairing.inflight
    assert len(pairing.journal) == 2
    assert pairing.acknowledge_final()


@pytest.mark.asyncio
async def test_one_role_is_the_same_rule_without_a_peer():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1)], events)
    pairing = _pairing(actor)

    assert _steps(pairing.submit_ready(), "actor") == [1]
    committed = await pairing.commit_staged_broadcasts()
    assert committed["actor"].applied_fragments == (0,)
    assert pairing.fragment_versions() == (1, 0)


def _sweep_candidate(global_step: int, fragment_id: int):
    return replace(
        _candidate("actor", global_step),
        permit=PullRequest(fragment_id, global_step, 1),
        local_step=1,
        c_steps=1,
    )


@pytest.mark.asyncio
async def test_dense_sweep_applies_only_a_complete_sweep():
    events = []
    actor = _PairedRole("actor", [_sweep_candidate(1, 0), _sweep_candidate(2, 1)], events,
                        release_broadcasts=False)
    pairing = _pairing(actor, sweep=True, pipeline_depth=1)

    assert _steps(pairing.submit_ready(), "actor") == [1, 2]  # whole sweep in flight
    actor._staged_steps.append(1)  # only fragment 0 broadcast so far
    actor._queued_steps.remove(1)
    committed = await pairing.commit_staged_broadcasts()
    assert committed["actor"].applied_fragments == ()
    assert actor.fragment_versions == (0, 0)

    actor.release_broadcasts = True
    committed = await pairing.commit_staged_broadcasts()
    assert committed["actor"].applied_fragments == (0, 1)
    assert actor.fragment_versions == (1, 2)


@pytest.mark.asyncio
async def test_dense_sweep_does_not_apply_while_only_part_of_the_sweep_is_permitted():
    actor = _PairedRole("actor", [_sweep_candidate(1, 0)], [])  # fragment 1 not PULLed yet
    pairing = _pairing(actor, sweep=True, pipeline_depth=1)

    assert _steps(pairing.submit_ready(), "actor") == [1]
    committed = await pairing.commit_staged_broadcasts()  # fragment 0 broadcast is staged

    assert committed["actor"].applied_fragments == ()
    assert actor.fragment_versions == (0, 0)


def test_sweep_profile_is_one_actor_with_horizon_one():
    with pytest.raises(ValueError, match="one actor role and H=1"):
        _profile(("actor", "critic"), sweep=True)
    with pytest.raises(ValueError, match="must come first"):
        _profile(("critic", "actor"))


def test_lockstep_requires_one_schedule_and_distinct_syncers():
    base = SimpleNamespace(syncer_addr=("h", 1), learner_id=0, learner_generation=0,
                           learner_generations=(0,), total_fragment_steps=4, expected_fragments=2,
                           local_horizon=2, pipeline_depth=2, training_contract_hash="a",
                           syncer_profile_hash="b")
    require_lockstep((base, SimpleNamespace(**{**vars(base), "syncer_addr": ("h", 2)})))
    with pytest.raises(ValueError, match="distinct syncer ports"):
        require_lockstep((base, base))
    with pytest.raises(ValueError, match="lockstep"):
        require_lockstep((base, SimpleNamespace(**{**vars(base), "syncer_addr": ("h", 2), "local_horizon": 3})))


class _Driver:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def _session(*streams, **profile):
    pairing = _pairing(*streams, **profile)
    return FullParameterSync(
        pairing,
        groups={stream.role: f"{stream.role}-group" for stream in streams},
        run=asyncio.run,
        trajectories=lambda rollout_id: f"evidence-{rollout_id}",
    )


def test_session_streams_one_round_and_publishes_the_actor_cut():
    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1)], events)
    critic = _PairedRole("critic", [_candidate("critic", 1)], events)
    session, driver = _session(actor, critic), _Driver()

    start = session.start(driver)
    assert (start.rollout_id, actor.group, critic.group) == (0, "actor-group", "critic-group")
    boundary = session.boundary(driver, rollout_id=0, stats=None)

    assert not boundary.stop
    assert boundary.state.policy_version == 1
    assert boundary.state.policy_tensor_hash() == "cut-(1, 0)"
    assert ("train-boundary", "critic", 0) in events
    name, fields = driver.events[-1]
    assert name == "rl_full_parameter_round"
    assert fields["sync/fragment_versions"] == [1, 0]


class _SlowBroadcastRole(_PairedRole):
    """Release broadcasts only after a few polls (a WAN round trip)."""

    def __init__(self, *args, polls_before_release, **kwargs):
        super().__init__(*args, release_broadcasts=False, **kwargs)
        self.polls_before_release = polls_before_release

    async def stage_available_broadcasts(self):
        self.polls_before_release -= 1
        self.release_broadcasts = self.polls_before_release <= 0
        return await super().stage_available_broadcasts()


def test_strict_sweep_boundary_blocks_until_the_sweep_is_applied():
    actor = _SlowBroadcastRole("actor", [_sweep_candidate(1, 0), _sweep_candidate(2, 1)], [],
                               polls_before_release=3)
    session, driver = _session(actor, sweep=True, pipeline_depth=1), _Driver()
    session.start(driver)

    boundary = session.boundary(driver, rollout_id=0, stats=None)

    assert actor.fragment_versions == (1, 2)
    assert boundary.state.policy_tensor_hash() == "cut-(1, 2)"
    assert not session.pairing.inflight


class _LatePullRole(_PairedRole):
    """The syncer permits the sweep's second fragment only after some polls."""

    def __init__(self, *args, late, polls_before_pull, **kwargs):
        super().__init__(*args, **kwargs)
        self.late = late
        self.polls_before_pull = polls_before_pull

    def ready_submission_candidates(self):
        self.polls_before_pull -= 1
        if self.polls_before_pull == 0 and self.late is not None:
            self.candidates.append(self.late)
            self.late = None
        return super().ready_submission_candidates()


def test_strict_sweep_boundary_waits_for_a_late_pull_of_the_sweep():
    actor = _LatePullRole("actor", [_sweep_candidate(1, 0)], [],
                          late=_sweep_candidate(2, 1), polls_before_pull=4)
    session, driver = _session(actor, sweep=True, pipeline_depth=1), _Driver()
    session.start(driver)

    boundary = session.boundary(driver, rollout_id=0, stats=None)

    assert actor.fragment_versions == (1, 2)
    assert boundary.state.policy_tensor_hash() == "cut-(1, 2)"


def test_strict_sweep_boundary_does_not_return_with_a_partial_sweep():
    actor = _LatePullRole("actor", [_sweep_candidate(1, 0)], [],
                          late=None, polls_before_pull=0)
    session, driver = _session(actor, sweep=True, pipeline_depth=1, wait_timeout=0.05), _Driver()
    session.start(driver)

    with pytest.raises(TimeoutError):
        session.boundary(driver, rollout_id=0, stats=None)
    assert actor.fragment_versions == (0, 0) and actor.closed


def test_terminal_round_acknowledges_every_role_after_publication():
    actor = _PairedRole("actor", [_candidate("actor", 3, base_version=1), _candidate("actor", 4, base_version=2)], [])
    critic = _PairedRole("critic", [_candidate("critic", 3, base_version=1), _candidate("critic", 4, base_version=2)], [])
    actor.fragment_versions = critic.fragment_versions = (1, 2)
    actor.finalizing = critic.finalizing = True
    session, driver = _session(actor, critic), _Driver()
    session.start(driver)

    boundary = session.boundary(driver, rollout_id=2, stats=None)
    assert boundary.stop and session.is_final_round(driver, rollout_id=2)
    with pytest.raises(RuntimeError, match="acknowledged its final cut"):
        session.finish(driver)
    session.published(driver, rollout_id=3, policy_hash="x")
    session.finish(driver)
    assert actor.final_acknowledged and critic.final_acknowledged


def test_close_closes_every_role_once_and_records_the_reason():
    actor = _PairedRole("actor", [], [])
    critic = _PairedRole("critic", [], [])
    session, driver = _session(actor, critic), _Driver()
    session.start(driver)

    session.close(LeaveReason.TRAINING_FAILED)
    session.close()

    assert actor.closed and critic.closed
    assert [e for e in driver.events if e[0] == "rl_sync_closed"] == [
        ("rl_sync_closed", {"sync/leave_reason": 5, "sync/leave_reason_name": "TRAINING_FAILED",
                            "sync/finished": False})
    ]


class _CloseFailsRole(_PairedRole):
    def close(self):
        super().close()
        raise OSError("socket already gone")


def test_close_closes_every_role_even_when_one_close_fails():
    actor = _CloseFailsRole("actor", [], [])
    critic = _PairedRole("critic", [], [])
    session, driver = _session(actor, critic), _Driver()
    session.start(driver)

    with pytest.raises(RuntimeError, match="actor close failed: OSError: socket already gone"):
        session.close(LeaveReason.PREEMPTED)

    assert actor.closed and critic.closed
    [(_, fields)] = [e for e in driver.events if e[0] == "rl_sync_closed"]
    assert (fields["sync/leave_reason"], fields["sync/leave_reason_name"]) == (3, "PREEMPTED")
    assert fields["sync/close_failures"] == ["actor close failed: OSError: socket already gone"]


def test_fragment_resident_state_never_materializes():
    from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState
    from yeto.rl.engine.trainable_state import UnsupportedLayoutError

    state = TrainerResidentState(
        layout="fragments",
        base_model_revision="r",
        config_hash="c",
        layout_hash="l",
        policy_version=1,
        digest=PolicyDigest("p" * 64, "q" * 64, 8, ()),
        materialize_fn=lambda _version: pytest.fail("fragments must not materialize"),
    )
    assert state.policy_tensor_hash() == "p" * 64
    assert state.payload_digest == ("q" * 64, 8)
    with pytest.raises(UnsupportedLayoutError):
        state.materialize()


def _role_stream_config(role: str, port: int, *, sweep: bool = False):
    from yeto.rl.adapters.miles.full_parameter.role_stream import RoleStreamConfig
    from yeto.rl.local_learner import ComponentIdentity

    return RoleStreamConfig(
        algorithm="grpo",
        role=role,
        component=ComponentIdentity(role, "a" * 40, "c" * 64),
        syncer_addr=("127.0.0.1", port),
        learner_id=0,
        learner_generation=0,
        learner_generations=(0, 0),
        total_fragment_steps=4,
        expected_fragments=2,
        expected_layout_hash="1" * 64,
        local_horizon=1,
        optimizer_steps_per_round=1,
        training_contract_hash="2" * 64,
        syncer_profile_hash="3" * 64,
        pipeline_depth=1,
        algorithm_spec_sha256="a1" * 32,
    )


def test_miles_build_sync_selects_full_parameter_sync_for_a_full_parameter_island():
    from yeto.rl.adapters.miles.entry import _policy_state_for, build_sync
    from yeto.rl.adapters.miles.full_parameter.session import (
        FullParameterSyncConfig,
        MilesFullParameterPolicyState,
    )

    config = FullParameterSyncConfig((_role_stream_config("actor", 30100),), sweep=True)
    args = SimpleNamespace(yeto_rl_full_parameter_config=config)
    runner = SimpleNamespace(run=asyncio.run)

    sync, progress = build_sync(args, yeto_policy_sync=True, runner=runner, groups={"actor": object()})

    assert isinstance(sync, FullParameterSync) and progress is None
    assert sync.pairing.profile.sweep and sync.pairing.profile.roles == ("actor",)
    assert isinstance(_policy_state_for(sync), MilesFullParameterPolicyState)
    with pytest.raises(ValueError, match="runner and trainer groups"):
        build_sync(args, yeto_policy_sync=True)
    with pytest.raises(ValueError, match="one actor role and H=1"):
        FullParameterSyncConfig(
            (_role_stream_config("actor", 30100), _role_stream_config("critic", 30101)), sweep=True
        )


def test_close_defaults_to_the_p5_finished_code():
    import inspect

    # P5 code table: 0 is FINISHED (LeaveReason.FINISHED once P5 merges).
    assert inspect.signature(FullParameterSync.close).parameters["reason"].default == 0


def test_layout_stream_role_splits_only_role_lane_algorithms():
    from yeto.rl.local_learner import layout_stream_role

    assert layout_stream_role("grpo", "actor") is None
    assert layout_stream_role("sao", "actor") == "actor"
    assert layout_stream_role("sao", "critic") == "critic"
    with pytest.raises(ValueError, match="no trainable role"):
        layout_stream_role("grpo", "critic")
    with pytest.raises(ValueError, match="no trainable role"):
        layout_stream_role("dpo", "actor")


@pytest.mark.parametrize(("algorithm", "role", "expected"), [
    ("grpo", "actor", None),
    ("sao", "critic", "critic"),
])
def test_role_stream_initializes_dense_with_the_plain_layout(monkeypatch, algorithm, role, expected):
    # P0-1.11 G3-fp probe: the dense (grpo) actor stream passed stream_role="actor",
    # which ParameterLayout.create refuses for non-SAO algorithms.
    from yeto.rl.adapters.miles.full_parameter import role_stream as module
    from yeto.rl.local_learner import ComponentIdentity, ParameterLayout, ParameterSpec

    seen = {}

    class Stop(Exception):
        pass

    async def fake_initialize(group, **kwargs):
        seen.update(kwargs)
        ParameterLayout.create(
            algorithm=kwargs["algorithm"],
            components=kwargs["components"],
            specs=(ParameterSpec(role, "w", (2,), "float32", 2, "tp0"),),
            num_fragments=1,
            fragment_strategy="owner_affine",
            stream_role=kwargs["stream_role"],
        )
        raise Stop

    monkeypatch.setattr(module.MilesChunkedFullParameterAdapter, "initialize", fake_initialize)
    config = replace(
        _role_stream_config(role, 30100),
        algorithm=algorithm,
        component=ComponentIdentity(role, "a" * 40, "c" * 64),
    )
    with pytest.raises(Stop):
        asyncio.run(module.MilesFullParameterRoleStream(config).initialize(object()))
    assert seen["stream_role"] == expected
