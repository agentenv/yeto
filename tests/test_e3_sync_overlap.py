"""S26 E3 item 3: sync/train overlap of the full-parameter sweep (CPU only).

A numeric fake role stream and an in-process strict-sweep syncer drive the
real ``FullParameterSync``/``RolePairing``: the island trains round k+1 while
sweep k is in flight and rebases ``theta <- G_k + (local - base)``. The tests
check that every round's progress reaches the global weights exactly once
(no double counting), that the trainer keeps its progress across a rebase,
and the launcher/island policy-age rule.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from test_rl_algorithm_provenance import (  # noqa: F401
    _fake_sky,
    _launcher_args,
    _no_modal_listing,
)

from yeto.protocol import PullRequest
from yeto.rl.adapters.miles.full_parameter.role_stream import RoleSubmissionCandidate
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)
from yeto.rl.engine.sync_overlap import rebase_values, required_policy_age

P = 2  # fragments
NUMEL = 4


class _SweepSyncer:
    """One-island strict sweep: PULLs a sweep once the previous one committed;
    ``G_f += pushed delta``; the terminal step publishes FINAL."""

    def __init__(self, initial: list[np.ndarray], sweeps: int) -> None:
        self.global_ = [value.copy() for value in initial]
        self.versions = [0] * P
        self.sweeps = sweeps
        self.step = 0  # last committed global step
        self.pushed: list[tuple[int, np.ndarray]] = []
        self.bcasts: list[tuple[int, int, np.ndarray]] = []
        self.finalizing = False
        self._since_commit = 0

    def permits(self) -> list[PullRequest]:
        sweep = self.step // P  # completed sweeps
        if self.finalizing or sweep >= self.sweeps or self.step % P:
            return []
        pushed = {step for step, _ in self.pushed}
        return [PullRequest(f, sweep * P + f + 1, 1) for f in range(P) if sweep * P + f + 1 not in pushed]

    def push(self, step: int, fragment_id: int, delta: np.ndarray) -> None:
        assert self.step < step <= self.step + P and (step - 1) % P == fragment_id
        self.pushed.append((step, delta.copy()))
        self.global_[fragment_id] = self.global_[fragment_id] + delta
        self.versions[fragment_id] = step
        self._since_commit += 1
        if self._since_commit < P:
            return
        self._since_commit = 0
        self.step += P
        if self.step == self.sweeps * P:
            self.finalizing = True
        else:
            self.bcasts.extend((f, self.versions[f], self.global_[f].copy()) for f in range(P))


class _NumericRole:
    """A RoleStream whose cuts are lists of per-fragment f32 arrays."""

    role = "actor"

    def __init__(self, syncer: _SweepSyncer, progress) -> None:
        self.syncer = syncer
        self.progress = progress  # rollout_id -> list of per-fragment progress
        self.config = SimpleNamespace(overlap=True)
        self.current = [g.copy() for g in syncer.global_]
        self.anchors = [g.copy() for g in syncer.global_]
        self._versions = [0] * P
        self.submission = None
        self.submission_versions = None
        self.local_rounds = self.submission_rounds = 0
        self.rounds_at_anchor = [0] * P
        self.round_base = None
        self.rounds_at_base = 0
        self._pending: dict[int, PullRequest] = {}
        self._staged: dict[tuple[int, int], np.ndarray] = {}
        self.terminal_submitted = False
        self.final_acknowledged = False
        self.closed = False
        self.trained_from: list[list[np.ndarray]] = []

    @property
    def fragment_versions(self):
        return tuple(self._versions)

    @property
    def finalizing(self):
        return self.syncer.finalizing

    async def initialize(self, group):
        del group

    def resident_state(self, policy_version):
        return SimpleNamespace(policy_version=policy_version, policy_tensor_hash=lambda: "cut")

    async def complete_local_round(self, *, rollout_id, trajectories, hold_submission=False):
        del trajectories
        self.trained_from.append([c.copy() for c in self.current])
        self.current = [c + d for c, d in zip(self.current, self.progress(rollout_id), strict=True)]
        self.local_rounds += 1
        if not hold_submission:
            self.submission = [c.copy() for c in self.current]
            self.submission_versions = self.fragment_versions
            self.submission_rounds = self.local_rounds

    def ready_submission_candidates(self):
        for permit in self.syncer.permits():
            self._pending.setdefault(permit.fragment_id, permit)
        out = []
        for f, permit in sorted(self._pending.items()):
            base = max(0, permit.global_step - P)
            if (self.submission is None or self._versions[f] != base
                    or self.submission_versions[f] != base
                    or self.submission_rounds - self.rounds_at_anchor[f] < 1):
                continue
            out.append(RoleSubmissionCandidate(
                "actor", permit, base, self.submission_rounds, 1, 1, NUMEL * 4,
                self.submission, self.anchors[f]))
        return tuple(out)

    def retain_candidate(self, candidate):
        del candidate

    def release_candidate(self, candidate):
        del candidate

    def push_candidate(self, candidate):
        f = candidate.permit.fragment_id
        self.syncer.push(candidate.permit.global_step, f,
                         candidate.submission_cut[f] - candidate.anchor)
        self._pending.pop(f)
        self.terminal_submitted |= candidate.permit.global_step == self.syncer.sweeps * P
        return SimpleNamespace(role="actor", fragment_id=f, global_step=candidate.permit.global_step)

    async def stage_available_broadcasts(self):
        for f, version, value in self.syncer.bcasts:
            if version > self._versions[f]:
                self._staged[(f, version)] = value
        self.syncer.bcasts.clear()
        return tuple(self._staged)

    @property
    def staged_fragment_versions(self):
        return tuple(sorted(self._staged))

    def has_staged_broadcast(self, fragment_id, version):
        return (fragment_id, version) in self._staged

    async def apply_staged_broadcast(self, fragment_id, version, *, rebase=False):
        assert rebase, "overlap applies every sweep by rebasing"
        g = self._staged.pop((fragment_id, version))
        self.current[fragment_id] = rebase_values(
            g, self.current[fragment_id], self.round_base[fragment_id])
        self.anchors[fragment_id] = g
        self._versions[fragment_id] = version
        self.rounds_at_anchor[fragment_id] = self.rounds_at_base
        return (fragment_id,)

    def finish_rebase(self):
        self.submission = [c.copy() for c in self.current]
        self.submission_versions = self.fragment_versions
        self.submission_rounds = self.local_rounds

    def mark_round_base(self):
        self.round_base = [c.copy() for c in self.current]
        self.rounds_at_base = self.local_rounds

    async def wait_and_apply_final(self):
        self.current = [g.copy() for g in self.syncer.global_]
        self._versions = [self.syncer.sweeps * P - P + f + 1 for f in range(P)]

    def acknowledge_final(self):
        self.final_acknowledged = True

    def discard_detached_submission_cut(self):
        pass

    def close(self):
        self.closed = True


class _Driver:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def _run(sweeps: int, overlap: bool = True):
    rng = np.random.default_rng(7)
    initial = [rng.standard_normal(NUMEL).astype(np.float32) for _ in range(P)]
    steps = {k: [rng.standard_normal(NUMEL).astype(np.float32) * 0.1 for _ in range(P)]
             for k in range(sweeps)}
    syncer = _SweepSyncer(initial, sweeps)
    role = _NumericRole(syncer, lambda rollout_id: steps[rollout_id])
    profile = FullParameterSyncProfile(roles=("actor",), total_fragment_steps=sweeps * P, fragments=P,
                                       local_horizon=1, pipeline_depth=1, wait_timeout=5.0,
                                       poll_seconds=0.001, sweep=True, overlap=overlap)
    session = FullParameterSync(RolePairing({"actor": role}, profile), groups={"actor": "g"},
                                run=asyncio.run, trajectories=lambda rollout_id: rollout_id)
    driver = _Driver()
    session.start(driver)
    stops = []
    for rollout_id in range(sweeps):
        boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
        stops.append(boundary.stop)
        if boundary.stop:
            session.published(driver, rollout_id=rollout_id, policy_hash="x")
            break
    return initial, steps, syncer, role, stops


def test_overlap_counts_every_round_exactly_once():
    sweeps = 5
    initial, steps, syncer, role, stops = _run(sweeps)
    assert stops == [False] * (sweeps - 1) + [True]
    # Pushed deltas are exactly the per-round progress, in round order.
    pushed = sorted(syncer.pushed, key=lambda row: row[0])
    for k in range(sweeps):
        for f in range(P):
            np.testing.assert_allclose(pushed[k * P + f][1], steps[k][f], rtol=0, atol=1e-6)
    # Global = initial + sum of every round's progress, once.
    for f in range(P):
        want = initial[f] + sum(steps[k][f] for k in range(sweeps))
        np.testing.assert_allclose(syncer.global_[f], want, rtol=0, atol=1e-5)
        np.testing.assert_allclose(role.current[f], want, rtol=0, atol=1e-5)
    assert role.final_acknowledged


def test_overlap_trains_one_outer_step_stale_and_keeps_local_progress():
    sweeps = 4
    initial, steps, _syncer, role, _ = _run(sweeps)
    # Round 0 starts from the initial cut, round 1 from its own round-0 result
    # (sweep 0 still in flight), round k>=2 from G_{k-2} + progress of round k-1.
    for f in range(P):
        np.testing.assert_allclose(role.trained_from[0][f], initial[f])
        np.testing.assert_allclose(role.trained_from[1][f], initial[f] + steps[0][f], atol=1e-6)
        for k in range(2, sweeps):
            g_prev = initial[f] + sum(steps[j][f] for j in range(k - 1))
            np.testing.assert_allclose(role.trained_from[k][f], g_prev + steps[k - 1][f], atol=1e-5)


def test_rebase_values_is_anchor_plus_local_progress():
    anchor = np.array([1.0, -2.0], dtype=np.float32)
    current = np.array([0.5, 0.25], dtype=np.float32)
    base = np.array([0.25, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(rebase_values(anchor, current, base), [1.25, -2.25])
    with pytest.raises(ValueError, match="float32"):
        rebase_values(anchor.astype(np.float64), current, base)
    with pytest.raises(ValueError, match="non-finite"):
        rebase_values(np.array([np.inf, 0], np.float32), current, base)


def test_overlap_profile_needs_the_strict_sweep():
    with pytest.raises(ValueError, match="overlap needs the dense strict sweep"):
        FullParameterSyncProfile(roles=("actor",), total_fragment_steps=4, fragments=2, local_horizon=2,
                                 pipeline_depth=1, wait_timeout=1.0, poll_seconds=0.01, overlap=True)


def _dense(*extra):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", "2", "--total-steps", "2", *extra), gpu="modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_overlap_raises_the_minimum_policy_age_and_reaches_the_island():
    from yeto.launch.rl_island_task import _rl_trainable_island_flags

    assert required_policy_age(0) == 1
    args = _dense("--rl-sync-overlap")
    assert args.rl_max_policy_age == 1
    assert "--sweep-overlap" in _rl_trainable_island_flags(args, 0, 1)
    assert _dense("--rl-sync-overlap", "--rl-max-policy-age", "3").rl_max_policy_age == 3
    plain = _dense()
    assert int(plain.rl_max_policy_age or 0) == 0
    assert "--sweep-overlap" not in _rl_trainable_island_flags(plain, 0, 1)


def test_overlap_launch_fails_when_the_algorithm_spec_does_not_tolerate_the_age():
    from yeto.launch_preflight import PreflightError, check_policy_age_spec

    args = _dense("--rl-sync-overlap")
    with pytest.raises(PreflightError, match="--rl-max-policy-age 1 needs an algorithm spec"):
        check_policy_age_spec(args)


def test_island_refuses_overlap_without_the_raised_policy_age():
    from yeto.rl.adapters.miles.island_entry import _dense_full_sync_config

    args = SimpleNamespace(sweep_overlap=True, rl_max_policy_age=0)
    with pytest.raises(ValueError, match="--sweep-overlap .* needs --rl-max-policy-age >= 1"):
        _dense_full_sync_config(args, model_config_sha256="0" * 64,
                                training_contract_sha256="0" * 64, algorithm_spec_sha256="0" * 64)


def test_overlap_is_bound_into_the_role_stream_contract():
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config

    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        role_stream_session_contract_hash,
    )

    exact = _role_stream_config("actor", 30100, sweep=True)
    assert (role_stream_session_contract_hash("1" * 64, exact)
            != role_stream_session_contract_hash("1" * 64, replace(exact, overlap=True)))
