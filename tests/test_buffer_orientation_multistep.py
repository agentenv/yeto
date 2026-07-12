import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_buffer_orientation_multistep",
    ROOT / "scripts" / "replay_buffer_orientation_multistep.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


MU = 0.9
LR = 0.175


def _generator(seed: int = 7) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _case(dim: int = 64, commits: int = 5, seed: int = 3):
    generator = _generator(seed)
    current = torch.randn(dim, generator=generator)
    buffer = torch.randn(dim, generator=generator)
    deltas = [torch.randn(dim, generator=generator) for _ in range(commits)]
    return current, buffer, deltas


def _closed_form(current, buffer, deltas, lr, mu):
    """Direct evaluation of the unrolled recursion.

    b_k = mu^k b_0 + sum_{j=1..k} mu^(k-j) g_j
    theta_N = theta_0 - lr * sum_{k=1..N} (g_k + mu * b_k)
    """

    buffers = []
    for k in range(1, len(deltas) + 1):
        b_k = mu**k * buffer
        for j in range(1, k + 1):
            b_k = b_k + mu ** (k - j) * deltas[j - 1]
        buffers.append(b_k)
    theta = current.clone()
    for g_k, b_k in zip(deltas, buffers):
        theta = theta - lr * (g_k + mu * b_k)
    return theta, buffers[-1]


def test_single_commit_rollout_bit_matches_the_production_trial():
    current, buffer, deltas = _case(commits=1)
    merged_update = -deltas[0]

    theta, final_buffer, stats, snapshots = MOD.rollout(
        current, buffer, deltas, LR, MU
    )
    trial = MOD.bn._nesterov_trial(current, buffer, merged_update, LR, MU)

    assert torch.equal(theta, trial)
    assert torch.equal(final_buffer, buffer.mul(MU).add(deltas[0]))
    assert len(stats) == 1
    assert snapshots == {}


def test_rollout_matches_the_closed_form_recursion():
    current, buffer, deltas = _case(commits=6, seed=11)

    theta, final_buffer, stats, _ = MOD.rollout(current, buffer, deltas, LR, MU)
    expected_theta, expected_buffer = _closed_form(current, buffer, deltas, LR, MU)

    assert torch.allclose(theta, expected_theta, atol=1e-5)
    assert torch.allclose(final_buffer, expected_buffer, atol=1e-5)
    assert len(stats) == 6


def test_rollout_does_not_mutate_its_inputs():
    current, buffer, deltas = _case(commits=4, seed=17)
    current_copy = current.clone()
    buffer_copy = buffer.clone()
    delta_copies = [delta.clone() for delta in deltas]

    MOD.rollout(current, buffer, deltas, LR, MU, record_at=(1, 4))

    assert torch.equal(current, current_copy)
    assert torch.equal(buffer, buffer_copy)
    for delta, copy in zip(deltas, delta_copies):
        assert torch.equal(delta, copy)


def test_rollout_snapshots_match_truncated_rollouts():
    current, buffer, deltas = _case(commits=5, seed=23)

    _, _, _, snapshots = MOD.rollout(
        current, buffer, deltas, LR, MU, record_at=(1, 3, 5)
    )

    for k in (1, 3, 5):
        theta_k, _, _, _ = MOD.rollout(current, buffer, deltas[:k], LR, MU)
        assert torch.equal(snapshots[k], theta_k), k


def test_rollout_stats_track_the_exact_per_commit_geometry():
    current, buffer, deltas = _case(commits=3, seed=29)

    _, _, stats, _ = MOD.rollout(current, buffer, deltas, LR, MU)

    b = buffer.clone()
    theta = current.clone()
    cumulative = 0.0
    for k, delta in enumerate(deltas, start=1):
        delta_norm = float(delta.norm())
        c_k = float(torch.dot(b, delta)) / delta_norm**2
        entry = stats[k - 1]
        assert entry["commit"] == k
        assert math.isclose(entry["delta_norm"], delta_norm, rel_tol=1e-6)
        assert math.isclose(entry["buffer_norm"], float(b.norm()), rel_tol=1e-6)
        assert math.isclose(entry["c"], c_k, rel_tol=1e-6)
        assert math.isclose(
            entry["aligned_gain"], 1.0 + MU + MU * MU * c_k, rel_tol=1e-6
        )
        b = b.mul(MU).add(delta)
        next_theta = theta - LR * (delta + MU * b)
        step_norm = float((next_theta - theta).norm())
        cumulative += step_norm**2
        theta = next_theta
        assert math.isclose(entry["step_norm"], step_norm, rel_tol=1e-6)
        assert math.isclose(entry["cumulative_step_sq"], cumulative, rel_tol=1e-6)
        assert math.isclose(
            entry["displacement_norm"], float((theta - current).norm()), rel_tol=1e-6
        )


def test_aligned_buffer_compounds_more_displacement_than_anti_aligned():
    dim = 128
    delta = torch.randn(dim, generator=_generator(31))
    delta = delta / float(delta.norm())
    current = torch.zeros(dim)
    deltas = [delta.clone() for _ in range(8)]
    norm = 2.0

    aligned, _, _, _ = MOD.rollout(current, norm * delta, deltas, LR, MU)
    anti, _, _, _ = MOD.rollout(current, -norm * delta, deltas, LR, MU)
    zero, _, _, _ = MOD.rollout(current, torch.zeros(dim), deltas, LR, MU)

    assert float(aligned.norm()) > float(zero.norm()) > float(anti.norm())
    # With a constant delta the whole trajectory stays on the delta line, so
    # the aligned/anti-aligned displacement gap is exactly twice the buffer
    # contribution 2 * lr * norm * mu^2 * sum_k mu^(k-1).
    gap = float(aligned.norm()) - float(anti.norm())
    expected = (
        2.0 * LR * norm * MU * MU * sum(MU ** (k - 1) for k in range(1, 9))
    )
    assert math.isclose(gap, expected, rel_tol=1e-4)


def test_rollout_rejects_degenerate_inputs():
    current, buffer, deltas = _case(commits=2)
    with pytest.raises(ValueError, match="at least one delta"):
        MOD.rollout(current, buffer, [], LR, MU)
    with pytest.raises(ValueError, match="record_at"):
        MOD.rollout(current, buffer, deltas, LR, MU, record_at=(3,))
    with pytest.raises(ValueError, match="shape mismatch"):
        MOD.rollout(current, buffer, [deltas[0], deltas[1][:-1]], LR, MU)
    with pytest.raises(ValueError, match="numerically zero"):
        MOD.rollout(current, buffer, [torch.zeros_like(buffer)], LR, MU)
    with pytest.raises(ValueError, match="rank-1"):
        MOD.rollout(current, buffer[:-1], deltas, LR, MU)


def _fake_groups(count: int, fragments: int = 2):
    return [
        [{"fragment": index % fragments, "step": index + 1}] for index in range(count)
    ]


def test_select_branch_groups_requires_history_and_full_rollouts():
    groups = _fake_groups(40, fragments=2)

    descriptors = MOD.select_branch_groups(groups, 6, 3, 8)

    assert len(descriptors) == 6
    branch_indices = [descriptor["branch_index"] for descriptor in descriptors]
    assert branch_indices == sorted(set(branch_indices))
    for descriptor in descriptors:
        window = descriptor["rollout_indices"]
        assert len(window) == 8
        assert window[0] == descriptor["branch_index"]
        fragment = groups[window[0]][0]["fragment"]
        # Consecutive same-fragment commits with nothing skipped.
        assert [groups[index][0]["fragment"] for index in window] == [fragment] * 8
        assert all(right - left == 2 for left, right in zip(window, window[1:]))
        # At least three prior rounds for the fragment.
        assert window[0] >= 3 * 2
        # The rollout never runs past the capture.
        assert window[-1] < len(groups)


def test_select_branch_groups_returns_all_eligible_when_few():
    groups = _fake_groups(22, fragments=2)

    descriptors = MOD.select_branch_groups(groups, 10, 1, 8)

    # Fragment sequences have 11 commits each; positions 1..3 are eligible.
    assert len(descriptors) == 6
    with pytest.raises(ValueError, match="subsequent rounds"):
        MOD.select_branch_groups(_fake_groups(8, fragments=2), 6, 1, 8)
    with pytest.raises(ValueError, match="positive"):
        MOD.select_branch_groups(groups, 0, 1, 8)


def test_rank_correlation():
    assert math.isclose(
        MOD.rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0
    )
    assert math.isclose(
        MOD.rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0
    )
    assert MOD.rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    tied = MOD.rank_correlation([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0])
    assert 0.0 < tied < 1.0
    with pytest.raises(ValueError, match="equal-length"):
        MOD.rank_correlation([1.0], [1.0, 2.0])


def test_variant_helpers_are_shared_with_the_one_step_script():
    assert MOD.VARIANTS == MOD.orient.VARIANTS
    assert MOD.BASELINE_VARIANT == MOD.orient.BASELINE_VARIANT
    buffer, delta = (
        torch.randn(64, generator=_generator(41)),
        torch.randn(64, generator=_generator(43)),
    )
    variants = MOD.orient.build_buffer_variants(buffer, delta, _generator(47))
    for name, variant in variants.items():
        assert math.isclose(
            float(variant.norm()), float(buffer.norm()), rel_tol=1e-6
        ), name
