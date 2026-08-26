from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.protocol import PartialMessageGenerationLost, PullRequest
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.miles import get_current_published_policy_identity
from yeto.rl.miles_sao_streaming import (
    MilesFullParameterRoleStream,
    MilesSaoRoleStreamConfig,
    MilesSaoStreamingConfig,
    MilesSaoStreamingPolicySync,
    RoleRoundResult,
    _RoleSubmissionCandidate,
    sao_role_stream_session_contract_hash,
)


class _Remote:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        value = self.function(*args, **kwargs)
        if hasattr(value, "__await__"):
            return await value
        return value


def _role_config(
    role: str,
    *,
    port: int,
    fragments: int = 2,
    total_steps: int = 4,
) -> MilesSaoRoleStreamConfig:
    return MilesSaoRoleStreamConfig(
        role=role,
        component=ComponentIdentity(
            role, ("a" if role == "actor" else "b") * 40, "c" * 64
        ),
        syncer_addr=("127.0.0.1", port),
        learner_id=0,
        learner_generation=7,
        learner_generations=(7, 9),
        total_fragment_steps=total_steps,
        expected_fragments=fragments,
        expected_layout_hash=("d" if role == "actor" else "e") * 64,
        local_horizon=2,
        optimizer_steps_per_round=1 if role == "actor" else 2,
        training_contract_hash="f" * 64,
        syncer_profile_hash="0" * 64,
        pipeline_depth=2,
        wan_streams=2,
        wait_timeout=1,
        poll_seconds=0.001,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
    )


def test_role_sessions_are_domain_separated_and_dual_config_is_lockstep():
    actor = _role_config("actor", port=30100)
    critic = _role_config("critic", port=30101)
    config = MilesSaoStreamingConfig(actor, critic)

    assert config.actor.role == "actor"
    assert sao_role_stream_session_contract_hash("1" * 64, actor) != (
        sao_role_stream_session_contract_hash("1" * 64, critic)
    )
    assert sao_role_stream_session_contract_hash("1" * 64, actor) != (
        sao_role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, syncer_profile_hash="2" * 64),
        )
    )
    assert sao_role_stream_session_contract_hash("1" * 64, actor) != (
        sao_role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, pipeline_depth=1),
        )
    )
    assert sao_role_stream_session_contract_hash("1" * 64, critic) != (
        sao_role_stream_session_contract_hash(
            "1" * 64,
            replace(critic, optimizer_steps_per_round=3),
        )
    )
    with pytest.raises(ValueError, match="distinct syncer ports"):
        MilesSaoStreamingConfig(actor, _role_config("critic", port=30100))
    with pytest.raises(ValueError, match="lockstep"):
        MilesSaoStreamingConfig(
            actor,
            _role_config("critic", port=30101, fragments=3, total_steps=6),
        )
    with pytest.raises(ValueError, match="lockstep"):
        MilesSaoStreamingConfig(
            actor,
            replace(critic, pipeline_depth=1),
        )


class _Client:
    def __init__(self, pulls):
        self.pulls = list(pulls)
        self.pushes = []
        self.connection_generation = 1

    def check_health(self):
        return None

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls

    def push_fragment_parts(self, *args):
        *metadata, parts = args
        self.pushes.append((tuple(metadata), b"".join(parts)))
        return True


def test_role_stream_uses_sao_round_horizon_not_critic_adam_update_count():
    stream = MilesFullParameterRoleStream(_role_config("critic", port=30101))
    stream.client = _Client([PullRequest(0, 1, 1)])
    stream.adapter = SimpleNamespace(
        layout=SimpleNamespace(
            fragments=SimpleNamespace(fragments=[SimpleNamespace(numel=2)])
        ),
        delta_parts_from_authoritative=lambda *_args, **_kwargs: (
            lambda: (b"\0\0\0\0\0\0\0\0",)
        ),
    )
    stream.current = object()
    stream.submission_cut = stream.current
    stream.submission_fragment_versions = (0,)
    stream.local_rounds = 2
    stream.action_tokens = 11
    stream.submission_rounds = 2
    stream.submission_tokens = 11
    stream._anchors = [object()]
    stream._fragment_versions = [0]
    stream._rounds_at_anchor = [0]
    stream._tokens_at_anchor = [0]

    candidates = stream.ready_submission_candidates()
    stream.retain_candidate(candidates[0])
    submissions = tuple(
        submission
        for candidate in candidates
        if (submission := stream.push_candidate(candidate)) is not None
    )

    assert len(submissions) == 1
    assert submissions[0].c_steps == 2
    assert submissions[0].c_tokens == 11
    assert stream.client.pushes[0][0] == (0, 1, 1, 0, 2, 2, 11)
    assert stream.client.pushes[0][1] == b"\0" * 8

    # A reconnect may replay the same permit; it must be answered again from
    # the immutable current cut rather than suppressed as a duplicate from
    # the dead connection generation.
    stream.client.connection_generation = 2
    stream.client.pulls = [PullRequest(0, 1, 1)]
    candidates = stream.ready_submission_candidates()
    assert len(candidates) == 1
    assert stream.push_candidate(candidates[0]) is not None
    assert len(stream.client.pushes) == 2
    stream.release_candidate(candidates[0])


class _PairedRole:
    def __init__(
        self,
        role: str,
        candidates: list[_RoleSubmissionCandidate],
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
    return _RoleSubmissionCandidate(
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
async def test_paired_coordinator_nonblockingly_retains_asymmetric_pull():
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    sync = MilesSaoStreamingPolicySync(SimpleNamespace(), config)
    sync.actor = _PairedRole("actor", [_candidate("actor", 1)], events)
    sync.critic = _PairedRole(
        "critic",
        [_candidate("critic", 1)],
        events,
        empty_polls=2,
    )

    actor, critic = await sync._submit_paired_ready()
    assert actor == critic == ()
    assert events == []

    actor, critic = await sync._submit_paired_ready()
    assert actor == critic == ()
    actor, critic = await sync._submit_paired_ready()

    assert [value.global_step for value in actor] == [1]
    assert [value.global_step for value in critic] == [1]
    assert events == [("actor", 1, True), ("critic", 1, True)]


@pytest.mark.asyncio
async def test_pipeline_depth_two_queues_later_pair_without_wan_commit():
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    sync = MilesSaoStreamingPolicySync(SimpleNamespace(), config)
    sync.actor = _PairedRole(
        "actor",
        [_candidate("actor", 1), _candidate("actor", 2)],
        events,
    )
    sync.critic = _PairedRole(
        "critic",
        [_candidate("critic", 1), _candidate("critic", 2)],
        events,
        release_broadcasts=False,
    )
    sync.actor.release_broadcasts = False

    await sync._complete_local_rounds(rollout_id=0, trajectories=object())
    actor, critic = await sync._submit_paired_ready()
    assert [value.global_step for value in actor] == [1, 2]
    assert [value.global_step for value in critic] == [1, 2]
    assert len(sync._inflight_pairs) == 2
    assert await sync._commit_staged_paired_broadcasts() == ((), ())
    # A second Miles local-train boundary completes while both earlier WAN
    # pairs remain uncommitted: this is the intended streaming overlap.
    await sync._complete_local_rounds(rollout_id=1, trajectories=object())
    assert sync.actor.local_boundaries == sync.critic.local_boundaries == 2
    assert len(sync._inflight_pairs) == 2
    assert [event for event in events if event[0] in {"actor", "critic"}] == [
        ("actor", 1, True),
        ("critic", 1, True),
        ("actor", 2, True),
        ("critic", 2, True),
    ]


@pytest.mark.asyncio
async def test_pair_replays_a_true_enqueue_that_never_reaches_server_commit():
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    sync = MilesSaoStreamingPolicySync(SimpleNamespace(), config)
    actor_candidate = _candidate("actor", 1)
    sync.actor = _LostAfterQueueRole("actor", [actor_candidate], events)
    sync.critic = _PairedRole(
        "critic",
        [_candidate("critic", 1)],
        events,
    )

    await sync._submit_paired_ready()
    actor_applied, critic_applied = await sync._commit_staged_paired_broadcasts()
    assert actor_applied == ()
    assert critic_applied == ()
    await sync._submit_paired_ready()
    actor_applied, critic_applied = await sync._commit_staged_paired_broadcasts()

    assert actor_applied == (0,)
    assert critic_applied == (0,)
    assert sync.actor.fragment_versions == (1, 0)
    assert sync.critic.fragment_versions == (1, 0)
    assert events == [
        ("actor", 1, True),
        ("critic", 1, True),
        ("actor", 1, True),
    ]
    assert len(sync._paired_journal) == 1


@pytest.mark.asyncio
async def test_partial_push_replays_exact_cut_with_incremented_round_attempt():
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    sync = MilesSaoStreamingPolicySync(SimpleNamespace(), config)
    sync.actor = _PartialThenIncrementedAttemptRole(
        "actor",
        [_candidate("actor", 1)],
        events,
    )
    sync.critic = _PairedRole(
        "critic",
        [_candidate("critic", 1)],
        events,
    )

    actor, critic = await sync._submit_paired_ready()
    assert actor == ()
    assert [value.round_attempt for value in critic] == [1]
    actor, critic = await sync._submit_paired_ready()

    assert [value.round_attempt for value in actor] == [2]
    assert critic == ()
    assert events == [
        ("actor", 1, False),
        ("critic", 1, True),
        ("actor", 1, True),
    ]


@pytest.mark.asyncio
async def test_one_sided_timeout_fail_stops_without_releasing_later_step():
    events = []
    actor_config = replace(
        _role_config("actor", port=30100),
        wait_timeout=0.01,
    )
    critic_config = replace(
        _role_config("critic", port=30101),
        wait_timeout=0.01,
    )
    sync = MilesSaoStreamingPolicySync(
        SimpleNamespace(),
        MilesSaoStreamingConfig(actor_config, critic_config),
    )
    sync.actor = _PairedRole(
        "actor",
        [_candidate("actor", 1), _candidate("actor", 2)],
        events,
    )
    sync.critic = _PairedRole(
        "critic",
        [_candidate("critic", 1), _candidate("critic", 2)],
        events,
        false_pushes=10_000,
    )

    await sync._submit_paired_ready()
    await asyncio.sleep(0.02)
    with pytest.raises(TimeoutError, match="one-sided paired submission"):
        await sync._submit_paired_ready()

    assert ("actor", 2, True) in events
    assert len(sync._inflight_pairs) == 2
    with pytest.raises(RuntimeError, match="fail-stopped"):
        await sync._submit_paired_ready()


@pytest.mark.asyncio
async def test_terminal_final_manifest_drains_all_retained_pipeline_pairs():
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    sync = MilesSaoStreamingPolicySync(SimpleNamespace(), config)
    sync.actor = _PairedRole(
        "actor",
        [
            _candidate("actor", 3, base_version=1),
            _candidate("actor", 4, base_version=2),
        ],
        events,
        release_broadcasts=False,
    )
    sync.critic = _PairedRole(
        "critic",
        [
            _candidate("critic", 3, base_version=1),
            _candidate("critic", 4, base_version=2),
        ],
        events,
        release_broadcasts=False,
    )
    sync.actor.fragment_versions = sync.critic.fragment_versions = (1, 2)

    await sync._submit_paired_ready()
    assert sync._terminal_seen
    assert len(sync._inflight_pairs) == 2
    sync.actor.finalizing = sync.critic.finalizing = True
    actor_result, critic_result = await sync._finish_paired_without_more_training(
        RoleRoundResult("actor", object(), (), (), False),
        RoleRoundResult("critic", object(), (), (), False),
    )

    assert actor_result.terminal_submitted
    assert critic_result.terminal_submitted
    assert sync.actor.fragment_versions == sync.critic.fragment_versions == (3, 4)
    assert not sync._inflight_pairs
    assert len(sync._paired_journal) == 2


class _FakeRoleStream:
    def __init__(self, role: str, events: list[object]) -> None:
        self.role = role
        self.events = events
        self.hashes = [
            ("1" if role == "actor" else "8") * 64,
            ("2" if role == "actor" else "9") * 64,
            ("3" if role == "actor" else "a") * 64,
        ]
        self.round = 0
        self.fragment_versions = (0, 0)
        self.final_acknowledged = False
        self.fail_ack_once = False

    @property
    def content_hash(self):
        return self.hashes[self.round]

    @property
    def finalizing(self):
        return self.round == 2

    async def initialize(self, model):
        self.events.append(("initialize", self.role, model))

    async def complete_local_round(self, *, rollout_id, trajectories):
        self.round += 1
        self.fragment_versions = (self.round, self.round)
        terminal = self.round == 2
        self.events.append(("train-boundary", self.role, rollout_id))
        return RoleRoundResult(self.role, object(), (), (), terminal)

    def ready_submission_candidates(self):
        return ()

    async def stage_available_broadcasts(self):
        return ()

    @property
    def staged_fragment_versions(self):
        return ()

    def has_staged_broadcast(self, _fragment_id, _version):
        return False

    async def apply_staged_broadcast(self, _fragment_id, _version):
        raise AssertionError("fake role has no staged broadcast")

    async def wait_and_apply_final(self):
        self.events.append(("final-apply", self.role))

    def discard_detached_submission_cut(self):
        self.events.append(("discard-submission", self.role))

    def acknowledge_final(self):
        if self.fail_ack_once:
            self.fail_ack_once = False
            self.events.append(("final-ack-failed", self.role))
            raise RuntimeError(f"{self.role} ACK failed once")
        if self.final_acknowledged:
            self.events.append(("final-ack-replay", self.role))
            return
        self.events.append(("final-ack", self.role))
        self.final_acknowledged = True

    def close(self):
        self.events.append(("close", self.role))


@pytest.mark.asyncio
async def test_actor_and_critic_commit_together_but_only_actor_forms_rollout_identity(
    tmp_path,
):
    events = []
    config = MilesSaoStreamingConfig(
        _role_config("actor", port=30100),
        _role_config("critic", port=30101),
    )
    args = SimpleNamespace(
        start_rollout_id=0,
        yeto_rl_trajectory_evidence_dir=str(tmp_path),
        yeto_rl_event_tape=str(tmp_path / "events.jsonl"),
        yeto_rl_learner_id=0,
        yeto_rl_sync_preset="sao-streaming-full",
        wandb=False,
    )
    sync = MilesSaoStreamingPolicySync(args, config)
    sync.actor = _FakeRoleStream("actor", events)
    sync.critic = _FakeRoleStream("critic", events)
    sync._trajectory_evidence = lambda _rollout_id: object()
    sync._record_round = lambda pending: events.append(
        ("round", pending.rollout_id, pending.terminal)
    )
    sync._record_publication = lambda *, terminal: events.append(
        ("publication", sync.published_rollout_id, terminal)
    )
    actor_model = object()
    critic_model = object()
    versions = []
    identities = []
    engine = SimpleNamespace(
        update_weight_version=_Remote(
            lambda token: versions.append(token) or events.append(("token", token))
        )
    )
    rollout_manager = SimpleNamespace(
        set_external_policy_identity=_Remote(
            lambda version, policy_hash: (
                identities.append((version, policy_hash)) or (version, policy_hash)
            )
        ),
    )

    await sync.initialize(
        actor_model=actor_model,
        critic_model=critic_model,
        rollout_manager=rollout_manager,
    )
    publication_info = SimpleNamespace(rollout_engines=[engine])
    await sync.after_inference_publication(
        rollout_id=None,
        actor_model=actor_model,
        publication_info=publication_info,
    )
    assert get_current_published_policy_identity(args, expected_policy_version=0) == (
        0,
        "1" * 64,
    )

    assert not await sync.after_local_train(
        rollout_id=0,
        actor_model=actor_model,
        critic_model=critic_model,
        rollout_data=object(),
    )
    await sync.after_inference_publication(
        rollout_id=0,
        actor_model=actor_model,
        publication_info=publication_info,
    )
    assert versions[-1] == f"yeto:1:{'2' * 64}"
    assert "9" * 64 not in versions[-1]

    assert await sync.after_local_train(
        rollout_id=1,
        actor_model=actor_model,
        critic_model=critic_model,
        rollout_data=object(),
    )
    assert not any(event[0] == "final-ack" for event in events)
    sync.critic.fail_ack_once = True
    with pytest.raises(RuntimeError, match="terminal ACK publication failed"):
        await sync.after_inference_publication(
            rollout_id=1,
            actor_model=actor_model,
            publication_info=publication_info,
        )
    assert sync.actor.final_acknowledged
    assert not sync.critic.final_acknowledged
    assert sync.pending_publication is not None
    assert not sync.finished
    await sync.after_inference_publication(
        rollout_id=1,
        actor_model=actor_model,
        publication_info=publication_info,
    )
    assert identities[-1] == (2, "3" * 64)
    terminal_token_indices = [
        index
        for index, event in enumerate(events)
        if event[0] == "token" and event[1].startswith("yeto:2:")
    ]
    ack_indices = [
        index for index, event in enumerate(events) if event[0] == "final-ack"
    ]
    assert len(terminal_token_indices) == 2
    assert ack_indices and all(
        index > terminal_token_indices[0] for index in ack_indices
    )

    await sync.finalize()
    assert events[-2:] == [("close", "actor"), ("close", "critic")]
