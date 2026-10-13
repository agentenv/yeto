from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.protocol import PullRequest
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleStreamConfig,
    role_stream_session_contract_hash,
)
from yeto.rl.adapters.miles.full_parameter.session import FullParameterSyncConfig
from yeto.rl.engine.full_parameter_sync import FullParameterSync


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
        algorithm_spec_sha256="a1" * 32,
        wan_streams=2,
        wait_timeout=1,
        poll_seconds=0.001,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
    )


def test_role_sessions_are_domain_separated_and_dual_config_is_lockstep():
    actor = _role_config("actor", port=30100)
    critic = _role_config("critic", port=30101)
    config = FullParameterSyncConfig((actor, critic))

    assert tuple(config.by_role()) == ("actor", "critic")
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash("1" * 64, critic)
    )
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, syncer_profile_hash="2" * 64),
        )
    )
    # P0-4.8: a different algorithm spec is a different session.
    assert role_stream_session_contract_hash("1" * 64, actor) != (
        role_stream_session_contract_hash(
            "1" * 64,
            replace(actor, algorithm_spec_sha256="b2" * 32),
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
        FullParameterSyncConfig((actor, _role_config("critic", port=30100)))
    with pytest.raises(ValueError, match="lockstep"):
        FullParameterSyncConfig(
            (actor, _role_config("critic", port=30101, fragments=3, total_steps=6))
        )
    with pytest.raises(ValueError, match="lockstep"):
        FullParameterSyncConfig((actor, replace(critic, pipeline_depth=1)))


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


def test_sao_island_builds_the_two_role_streaming_full_parameter_sync():
    """SAO on ports is the shared full-parameter sync with actor and critic (P0-D8)."""

    from yeto.rl.adapters.miles.entry import build_sync

    config = FullParameterSyncConfig(
        (_role_config("actor", port=30100), _role_config("critic", port=30101))
    )
    args = SimpleNamespace(yeto_rl_full_parameter_config=config)
    groups = {"actor": object(), "critic": object()}

    sync, progress = build_sync(
        args, yeto_policy_sync=True, runner=SimpleNamespace(run=asyncio.run), groups=groups
    )

    assert isinstance(sync, FullParameterSync) and progress is None
    assert sync.pairing.profile.roles == ("actor", "critic")
    assert not sync.pairing.profile.sweep
    assert sync.pairing.profile.local_horizon == 2
    assert sync.actor is sync.pairing.streams["actor"]
    with pytest.raises(ValueError, match="exactly one trainer group"):
        build_sync(
            args,
            yeto_policy_sync=True,
            runner=SimpleNamespace(run=asyncio.run),
            groups={"actor": object()},
        )
