"""Torch-free checks for full SCAFFOLD Option II f32 algebra."""

import numpy as np


def _accumulate(controls, residuals, beta=1.0):
    mean = residuals.sum(axis=0, dtype=np.float32) / np.float32(len(residuals))
    return controls + np.float32(beta) * (residuals - mean)


def test_full_option_ii_keeps_the_persistent_bias_fixed_point():
    controls = np.zeros((4, 3), dtype=np.float32)
    residuals = np.array(
        [[1, 2, -4], [-1, -2, 4], [3, -1, 2], [-3, 1, -2]],
        dtype=np.float32,
    )
    first = _accumulate(controls, residuals)
    second = _accumulate(first, np.zeros_like(residuals))
    np.testing.assert_array_equal(second, first)


def test_full_option_ii_multiround_f32_zero_sum_is_tolerance_bounded():
    rng = np.random.default_rng(223223)
    controls = np.zeros((4, 256), dtype=np.float32)
    for _ in range(340):
        residuals = rng.normal(0.0, 1e-7, controls.shape).astype(np.float32)
        controls = _accumulate(controls, residuals)
    correction_sum = controls.sum(axis=0, dtype=np.float32)
    scale = float(np.max(np.abs(controls)))
    tolerance = max(2e-6, 2e-5 * scale)
    measured = float(np.max(np.abs(correction_sum)))
    print(f"full f32 zero-sum max_abs={measured:.9g} tol={tolerance:.9g}")
    assert measured <= tolerance


def test_identity_shuffle_deranges_canonical_controls_and_preserves_invariants():
    canonical = np.array(
        [[3, -2], [-3, 2], [4, 2], [-4, -2]], dtype=np.float32
    )
    shuffled = np.roll(canonical, -1, axis=0)
    assert all(
        not np.array_equal(own, assigned)
        for own, assigned in zip(canonical, shuffled)
    )
    np.testing.assert_array_equal(
        np.sort(np.linalg.norm(shuffled, axis=1)),
        np.sort(np.linalg.norm(canonical, axis=1)),
    )
    np.testing.assert_array_equal(canonical.sum(axis=0), np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(shuffled.sum(axis=0), np.zeros(2, dtype=np.float32))
