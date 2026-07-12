import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_buffer_orientation", ROOT / "scripts" / "replay_buffer_orientation.py"
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _case(dim: int = 512, seed: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    buffer = torch.randn(dim, generator=generator)
    delta = torch.randn(dim, generator=generator)
    return buffer, delta


def test_variants_share_the_exact_buffer_norm():
    buffer, delta = _case()
    variants = MOD.build_buffer_variants(buffer, delta, _generator())

    assert set(variants) == set(MOD.VARIANTS)
    for name, variant in variants.items():
        assert math.isclose(
            float(variant.norm()), float(buffer.norm()), rel_tol=1e-6
        ), name


def test_variant_orientations_are_correct():
    buffer, delta = _case()
    variants = MOD.build_buffer_variants(buffer, delta, _generator())

    assert torch.allclose(variants["real"], buffer)
    assert math.isclose(_cosine(variants["aligned"], delta), 1.0, abs_tol=1e-6)
    assert math.isclose(_cosine(variants["anti_aligned"], delta), -1.0, abs_tol=1e-6)
    assert abs(_cosine(variants["orthogonal"], delta)) < 1e-6
    # The orthogonalized variant stays in the span of the real buffer and the
    # delta direction: removing the delta projection recovers its direction.
    residual = buffer - torch.dot(buffer, delta) / torch.dot(delta, delta) * delta
    assert math.isclose(_cosine(variants["orthogonal"], residual), 1.0, abs_tol=1e-6)
    # A 512-dim random rotation is nearly orthogonal to everything but must
    # not be exactly the orthogonalized direction.
    assert abs(_cosine(variants["random_rotated"], delta)) < 0.2
    assert not torch.allclose(variants["random_rotated"], variants["orthogonal"])


def test_random_rotation_is_deterministic_per_seed_and_varies_across_seeds():
    buffer, delta = _case()

    first = MOD.build_buffer_variants(buffer, delta, _generator(11))
    second = MOD.build_buffer_variants(buffer, delta, _generator(11))
    third = MOD.build_buffer_variants(buffer, delta, _generator(12))

    assert torch.equal(first["random_rotated"], second["random_rotated"])
    assert not torch.allclose(first["random_rotated"], third["random_rotated"])


def test_parallel_buffer_uses_the_orthogonal_fallback_draw():
    delta = torch.randn(256, generator=_generator(5))
    buffer = 3.25 * delta

    variants = MOD.build_buffer_variants(buffer, delta, _generator(5))

    assert abs(_cosine(variants["orthogonal"], delta)) < 1e-6
    assert math.isclose(
        float(variants["orthogonal"].norm()), float(buffer.norm()), rel_tol=1e-6
    )


def test_degenerate_inputs_raise():
    buffer, delta = _case()
    with pytest.raises(ValueError, match="delta norm"):
        MOD.build_buffer_variants(buffer, torch.zeros_like(delta), _generator())
    with pytest.raises(ValueError, match="buffer norm"):
        MOD.build_buffer_variants(torch.zeros_like(buffer), delta, _generator())
    with pytest.raises(ValueError, match="rank-1"):
        MOD.build_buffer_variants(buffer, delta[:-1], _generator())


def test_buffer_geometry_matches_the_exact_decomposition():
    buffer, delta = _case()
    geometry = MOD.buffer_geometry(buffer, delta, MU)

    delta_norm = float(delta.norm())
    c_t = float(torch.dot(buffer, delta)) / delta_norm**2
    assert math.isclose(geometry["c_t"], c_t, rel_tol=1e-6)
    assert math.isclose(geometry["aligned_gain"], 1.0 + MU + MU * MU * c_t, rel_tol=1e-6)
    residual = buffer - c_t * delta
    assert math.isclose(
        geometry["transverse_ratio"],
        MU * MU * float(residual.norm()) / delta_norm,
        rel_tol=1e-6,
    )

    variants = MOD.build_buffer_variants(buffer, delta, _generator())
    buffer_norm = float(buffer.norm())
    aligned = MOD.buffer_geometry(variants["aligned"], delta, MU)
    anti = MOD.buffer_geometry(variants["anti_aligned"], delta, MU)
    orthogonal = MOD.buffer_geometry(variants["orthogonal"], delta, MU)
    assert math.isclose(
        aligned["aligned_gain"],
        1.0 + MU + MU * MU * buffer_norm / delta_norm,
        rel_tol=1e-6,
    )
    assert math.isclose(
        anti["aligned_gain"],
        1.0 + MU - MU * MU * buffer_norm / delta_norm,
        rel_tol=1e-6,
    )
    assert math.isclose(orthogonal["aligned_gain"], 1.0 + MU, abs_tol=1e-5)
    assert aligned["aligned_gain"] > orthogonal["aligned_gain"] > anti["aligned_gain"]


def test_nesterov_trial_matches_the_two_term_form():
    buffer, delta = _case(dim=128, seed=9)
    current = torch.randn(128, generator=_generator(1))
    merged_update = -delta

    trial = MOD.bn._nesterov_trial(current, buffer, merged_update, LR, MU)
    expected = current - LR * MOD.nesterov_direction(delta, buffer, MU)

    assert torch.allclose(trial, expected, atol=1e-6)


def test_variant_trials_differ_only_by_the_buffer_term():
    buffer, delta = _case(dim=128, seed=13)
    current = torch.randn(128, generator=_generator(2))
    merged_update = -delta
    variants = MOD.build_buffer_variants(buffer, delta, _generator(13))

    trials = {
        name: MOD.bn._nesterov_trial(current, variant, merged_update, LR, MU)
        for name, variant in variants.items()
    }
    for name in MOD.VARIANTS:
        difference = trials[name] - trials["real"]
        expected = -LR * MU * MU * (variants[name] - buffer)
        assert torch.allclose(difference, expected, atol=1e-6), name


def test_buffer_recursion_matches_the_closed_form():
    generator = _generator(21)
    deltas = [torch.randn(64, generator=generator) for _ in range(5)]

    buffer = torch.zeros(64)
    for delta in deltas:
        buffer = buffer.mul(MU).add(delta)

    closed_form = sum(
        MU ** (len(deltas) - 1 - index) * delta for index, delta in enumerate(deltas)
    )
    assert torch.allclose(buffer, closed_form, atol=1e-5)


def test_select_commit_groups_is_evenly_spaced_and_respects_history():
    groups = [
        [{"fragment": index % 2, "step": index + 1}] for index in range(42)
    ]

    selected = MOD.select_commit_groups(groups, 10, 3)

    # The first three rounds of each fragment (indices 0..5) are ineligible.
    assert len(selected) == 10
    assert selected == sorted(set(selected))
    assert min(selected) >= 6
    assert max(selected) == 41
    gaps = [right - left for left, right in zip(selected, selected[1:])]
    assert max(gaps) - min(gaps) <= 2

    few = MOD.select_commit_groups(groups[:8], 10, 3)
    assert few == [6, 7]

    with pytest.raises(ValueError, match="prior rounds"):
        MOD.select_commit_groups(groups[:4], 10, 3)


def test_variant_seed_is_deterministic_and_commit_specific():
    assert MOD.variant_seed(223, 40, 3) == MOD.variant_seed(223, 40, 3)
    seeds = {
        MOD.variant_seed(223, 40, 3),
        MOD.variant_seed(223, 40, 7),
        MOD.variant_seed(223, 52, 3),
        MOD.variant_seed(211, 40, 3),
    }
    assert len(seeds) == 4
    assert all(0 <= value < 2**63 for value in seeds)


def test_paired_panel_stats():
    stats = MOD.paired_panel_stats([1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.3, 1.2])

    assert math.isclose(stats["mean_gain"], 0.05, abs_tol=1e-12)
    assert stats["win_rate"] == 0.75
    assert stats["panels"] == 4
    assert stats["se"] > 0.0

    with pytest.raises(ValueError, match="equal length"):
        MOD.paired_panel_stats([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        MOD.paired_panel_stats([1.0, float("nan")], [1.0, 1.0])
