"""S27 C6: evaluation checkpoints of the asynchronous full-parameter syncer.

``--rl-sync-checkpoint-at-steps T1,T2,...`` becomes the syncer's
``--checkpoint-at-steps``: at each listed global step (one fragment merge,
the ``step=`` of the syncer log) the syncer writes the consistent cut to
``<checkpoint>.step-<t>`` on a background thread while rounds keep merging.
The list enters the semantic profile only when set, so every existing hash
and syncer command stays unchanged.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401 - fixtures
from test_rl_full_parameter_async import (
    ASYNC,
    FULL,
    P,
    TWO_ISLANDS,
    _base,
    _delta,
    _drive,
    _flag,
    _launcher_args,
    _nesterov_reference,
    _plan,
    needs_cargo,
    syncer_binary,  # noqa: F401 - fixture
)

from yeto.syncer_profile import SYNCER_SEMANTIC_PROFILE_SCHEMA, SyncerSemanticProfile

# The Rust vectors of server.rs eval_checkpoint_steps_append_to_the_profile_hash_like_python.
BASE_VECTOR = "bcbef730a6bc46e6c1901e90b6ce5a5382b035292863ed5cfea0285da408cafe"
EVAL_CHECKPOINTS_VECTOR = "2e395d5104892c355f6cd2609bcf3f5011a26407848fb805b5e262bf912584f3"

_BASE = {
    "schema": SYNCER_SEMANTIC_PROFILE_SCHEMA, "learners": 2, "quorum": 2, "grace_ms": 0,
    "grace_gamma": 0.8, "grace_tau": 2.0, "pipeline": 2, "min_round_interval_ms": 0,
    "sync_interval_steps": 24.0, "delta_correction": "none", "quorum_timeout_s": 900,
    "final_ack_timeout_s": 900, "total_steps": 8, "policy_sweep_fragments": None, "outer_lr": 0.7,
    "outer_momentum": 0.9, "checkpoint_enabled": True, "checkpoint_every": 8, "resume": True,
    "mark_final_checkpoint": True, "learner_budget_steps": None, "max_base_lag": 0,
    "learner_weight": "equal", "require_profile_binding": True,
}

# Default launcher shape: --total-steps 3 x --fragments 4 = 12 global steps.
EVAL = ("--rl-sync-checkpoint-every", "1000", "--rl-sync-checkpoint-at-steps", "3,7,11")


def test_the_eval_steps_hash_matches_the_rust_vector_and_no_steps_keep_the_base():
    assert SyncerSemanticProfile.from_mapping(_BASE).sha256 == BASE_VECTOR
    empty = SyncerSemanticProfile.from_mapping({**_BASE, "checkpoint_at_steps": []})
    assert empty.sha256 == BASE_VECTOR
    on = SyncerSemanticProfile.from_mapping({**_BASE, "checkpoint_at_steps": [3, 5]})
    assert on.checkpoint_at_steps == (3, 5)
    assert on.sha256 == EVAL_CHECKPOINTS_VECTOR


@pytest.mark.parametrize(("change", "message"), [
    ({"checkpoint_at_steps": [5, 3]}, "strictly increasing"),
    ({"checkpoint_at_steps": [3, 8]}, "below total_steps"),
    ({"checkpoint_at_steps": [0, 3]}, r"checkpoint_at_steps must be in \[1, "),
    ({"checkpoint_at_steps": 3}, "list of steps"),
    ({"checkpoint_at_steps": [3], "checkpoint_every": 1}, "checkpoint_every != 1"),
    ({"checkpoint_at_steps": [3], "checkpoint_enabled": False, "resume": False,
      "mark_final_checkpoint": False}, "requires checkpointing"),
])
def test_the_profile_refuses_eval_steps_the_syncer_refuses(change, message):
    with pytest.raises(ValueError, match=message):
        SyncerSemanticProfile.from_mapping({**_BASE, **change})


def test_eval_steps_reach_the_syncer_command_the_profile_and_the_dry_run():
    from yeto import launcher

    args, plan = _plan(ASYNC + EVAL)
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--checkpoint-at-steps") == "3,7,11"
    assert _flag(actor["command"], "--checkpoint-every") == "1000"
    profile = launcher.rl_syncer_profile(args, 2)
    assert profile.checkpoint_at_steps == (3, 7, 11)
    for island in plan["island_requests"]:
        assert _flag(island["learner_command"], "--syncer-profile-sha256") == profile.sha256
        assert "checkpoint-at-steps" not in island["learner_command"]  # the syncer's alone
    plain_args, plain = _plan(ASYNC + EVAL[:2])
    assert "--checkpoint-at-steps" not in plain["role_syncers"][0]["command"]
    assert launcher.rl_syncer_profile(plain_args, 2).checkpoint_at_steps == ()
    assert launcher.rl_syncer_profile(plain_args, 2).sha256 != profile.sha256


@pytest.mark.parametrize(("extra", "message"), [
    (FULL + EVAL, "--rl-sync-checkpoint-at-steps only with --rl-sync-full-parameter-rounds async"),
    (ASYNC + EVAL[2:], "--rl-sync-checkpoint-every above 1 with --rl-sync-checkpoint-at-steps"),
    (ASYNC + EVAL[:2] + ("--rl-sync-checkpoint-at-steps", "7,3"), r"strictly increasing within 1\.\.11"),
    (ASYNC + EVAL[:2] + ("--rl-sync-checkpoint-at-steps", "3,3"), r"strictly increasing within 1\.\.11"),
    (ASYNC + EVAL[:2] + ("--rl-sync-checkpoint-at-steps", "0,3"), r"strictly increasing within 1\.\.11"),
    (ASYNC + EVAL[:2] + ("--rl-sync-checkpoint-at-steps", "12"), "the final checkpoint holds the last one"),
    (ASYNC + EVAL + ("--syncer-shards", "2"), "no --syncer-shards above 1 with --rl-sync-checkpoint-at-steps"),
])
def test_eval_step_refusals(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def test_eval_steps_only_apply_to_the_full_parameter_preset():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", EVAL[2:], gpu="modal:2xh100")
    args.controller = "local"
    with pytest.raises(ValueError, match="--rl-sync-checkpoint-at-steps only applies to --rl-sync-preset full-parameter"):
        _prepare_rl_args(args)


def test_eval_steps_must_be_integers():
    with pytest.raises(SystemExit):
        _launcher_args("ports", ASYNC + ("--rl-sync-checkpoint-at-steps", "3,x"), gpu=TWO_ISLANDS)


# -- the real Rust syncer -----------------------------------------------------------

@pytest.fixture
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "eval-checkpoints-test-key")


_WRITTEN = re.compile(r"eval checkpoint written step=(\d+) path=(\S+) seconds=([0-9.]+) "
                      r"bytes=(\d+) copied_fragments=(\d+)")


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_eval_checkpoints_hold_the_cut_of_their_step_and_finish_before_the_final_one(
        syncer_binary, tmp_path):  # noqa: F811 - fixture
    from yeto.export import parse_checkpoint

    rounds: dict[int, dict] = {}

    def push(learner, permit):
        delta = _delta(learner, permit.global_step)
        rounds.setdefault(permit.global_step, {})[learner] = (1.0, delta)
        return _base(permit.global_step), 1, 7, delta

    # Two outer rounds per fragment: 2 x P = 4 global steps; copies at 2 and 3.
    _, log = _drive(syncer_binary, tmp_path, (
        "--rl-sync-checkpoint-every", "1000", "--rl-sync-checkpoint-at-steps", "2,3"),
        total_steps=2, push=push)
    assert sorted(rounds) == list(range(1, 2 * P + 1))
    checkpoint = tmp_path / "state.ckpt"
    written = {int(step): (path, int(size)) for step, path, _, size, _ in _WRITTEN.findall(log)}
    assert sorted(written) == [2, 3], log
    for step in (2, 3):
        path = tmp_path / f"state.ckpt.step-{step}"
        assert f"eval checkpoint started step={step} path={path}" in log
        assert written[step] == (str(path), path.stat().st_size)
        cut = parse_checkpoint(path)
        assert cut.global_step == step
        # Fragment p was last merged at the largest step <= t with (step-1) % P == p.
        assert [version for version, _, _ in cut.fragments] == [
            max(s for s in range(1, step + 1) if (s - 1) % P == fid) for fid in range(P)]
        expected = _nesterov_reference({s: r for s, r in rounds.items() if s <= step}, lr=0.7, mu=0.9)
        for fid, (_, params, _) in enumerate(cut.fragments):
            np.testing.assert_allclose(params.numpy(), expected[fid], rtol=0, atol=1e-6)
    # The last copy is durable before the final checkpoint is written.
    assert log.index("eval checkpoint written step=3") < log.index("final checkpoint written")
    assert parse_checkpoint(checkpoint).global_step == 2 * P
    assert not list(tmp_path.glob("*.tmp"))
