from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.protocol import PullRequest
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.adapters.miles.publish import (
    get_current_published_policy_identity,
    set_current_published_policy_identity,
)
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleRoundResult,
    RoleStreamConfig,
    role_stream_session_contract_hash,
)
from yeto.rl.adapters.miles.models.sao_streaming import (
    MilesSaoStreamingConfig,
    MilesSaoStreamingPolicySync,
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
) -> RoleStreamConfig:
    return RoleStreamConfig(
        algorithm="sao",
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
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash("1" * 64, critic)
    )
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, syncer_profile_hash="2" * 64),
        )
    )
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, pipeline_depth=1),
        )
    )
    assert role_stream_session_contract_hash("1" * 64, critic) != (
        role_stream_session_contract_hash(
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
    rollout_args = SimpleNamespace(yeto_rl_sync_preset="sao-streaming-full")
    engine = SimpleNamespace(
        update_weight_version=_Remote(
            lambda token: versions.append(token) or events.append(("token", token))
        )
    )

    def install_rollout_identity(version, policy_hash):
        identities.append((version, policy_hash))
        return set_current_published_policy_identity(
            rollout_args,
            policy_version=version,
            policy_hash=policy_hash,
        )

    rollout_manager = SimpleNamespace(
        set_external_policy_identity=_Remote(install_rollout_identity),
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
    assert rollout_args.yeto_rl_policy_token == f"yeto:0:{'1' * 64}"

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
    assert rollout_args.yeto_rl_policy_token == versions[-1]
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
