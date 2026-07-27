import json
import random
import struct

import pytest

from yeto.outer_lr_controller import (
    FactorialSurface,
    OracleSchedule,
    ScaleOutput,
    SpectralSketch,
    outer_lr_scale,
    planned_fragment_steps,
)


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _f64_bits(value: float) -> bytes:
    return struct.pack("!d", value)


def _surface(*, coefficients=(0.0, 0.0, 0.0)) -> FactorialSurface:
    return FactorialSurface.from_mapping(
        {
            "schema": "yeto.outer_lr_drift_surface.v1",
            "output": "log2_lr_multiplier",
            "features": [
                {"name": "u", "source": "mu", "scale": 0.9},
                {
                    "name": "q",
                    "source": "T_planned",
                    "center": 5.0,
                    "scale": 10.0,
                },
                {
                    "name": "h",
                    "source": "spectral.window",
                    "transform": "log2",
                    "center": 9.0,
                },
            ],
            "terms": [
                {"coefficient": coefficients[0], "powers": {"u": 1}},
                {
                    "coefficient": coefficients[1],
                    "powers": {"q": 1, "u": 1},
                },
                {
                    "coefficient": coefficients[2],
                    "powers": {"h": 1, "u": 1},
                },
            ],
            "drift_scale_bounds": {"min": 0.125, "max": 8.0},
        }
    )


def test_transient_mode_property_bit_matches_existing_bias_correction_rule():
    rng = random.Random(0xA9E_A7E)
    for _ in range(20_000):
        mu = _f32(rng.randrange(0, 999_999) / 1_000_000)
        t = rng.randrange(1, 100_001)
        t_planned = t + rng.randrange(0, 1_000)
        legacy = 1.0 / (1.0 - mu ** min(t + 1, 2_147_483_647))
        output = outer_lr_scale(
            "transient", mu=mu, t=t, t_planned=t_planned
        )
        assert _f64_bits(output.scale) == _f64_bits(legacy)
        assert _f64_bits(output.transient_scale) == _f64_bits(legacy)
        assert output.drift_scale is None


def test_zero_drift_property_bit_reduces_to_transient_without_spectral_input():
    surface = _surface()
    rng = random.Random(0xD21F7)
    for _ in range(10_000):
        mu = _f32(rng.randrange(0, 999_999) / 1_000_000)
        t_planned = rng.randrange(1, 257)
        t = rng.randrange(1, t_planned + 1)
        transient = outer_lr_scale(
            "transient", mu=mu, t=t, t_planned=t_planned
        )
        measured = outer_lr_scale(
            "measured-drift",
            mu=mu,
            t=t,
            t_planned=t_planned,
            surface=surface,
        )
        assert _f64_bits(measured.scale) == _f64_bits(transient.scale)
        assert measured.drift_scale == 1.0


def test_measured_drift_uses_factorial_surface_and_spectral_sketch():
    output = outer_lr_scale(
        "measured-drift",
        mu=0.9,
        t=3,
        t_planned=10,
        surface=_surface(coefficients=(-0.2, -0.4, -0.1)),
        spectral_sketch={"window": 1024.0},
    )
    expected_drift = 2.0 ** (-0.2 - 0.4 * 0.5 - 0.1)
    expected_transient = 1.0 / (1.0 - 0.9**4)
    assert output.drift_scale == pytest.approx(expected_drift, abs=1e-15)
    assert output.transient_scale == expected_transient
    assert output.scale == pytest.approx(expected_transient * expected_drift)


def test_drift_bounds_apply_before_exponentiation():
    surface = FactorialSurface.from_mapping(
        {
            "schema": "yeto.outer_lr_drift_surface.v1",
            "output": "log2_lr_multiplier",
            "features": [],
            "terms": [{"coefficient": 1e300, "powers": {}}],
            "drift_scale_bounds": {"min": 0.125, "max": 8.0},
        }
    )
    assert outer_lr_scale(
        "measured-drift", mu=0.0, t=1, t_planned=1, surface=surface
    ) == ScaleOutput(scale=8.0, transient_scale=1.0, drift_scale=8.0)


def test_oracle_mode_is_exact_passthrough():
    for scale in (float.fromhex("0x1.0000000000001p-4"), 1.0, 3.75):
        assert outer_lr_scale(
            "oracle",
            mu=float("nan"),
            t=2,
            t_planned=4,
            oracle_scale=scale,
        ) == ScaleOutput(scale=scale, transient_scale=None, drift_scale=None)


def test_probe_sketch_overlays_fragment_features():
    sketch = SpectralSketch.from_mapping(
        {
            "schema": "yeto.outer_lr_spectral_sketch.v1",
            "global_features": {"lambda_max": 4.0, "trace": 10.0},
            "fragment_features": {"2": {"lambda_max": 7.0}},
        }
    )
    assert sketch.for_fragment(0) == {"lambda_max": 4.0, "trace": 10.0}
    assert sketch.for_fragment(2) == {"lambda_max": 7.0, "trace": 10.0}


def test_oracle_schedule_is_one_indexed_and_requires_complete_horizon():
    oracle = OracleSchedule.from_mapping(
        {
            "schema": "yeto.outer_lr_oracle_schedule.v1",
            "scales": [1.0, 0.75],
            "fragment_scales": {"0": [2.0, 1.5, 1.0]},
        }
    )
    assert oracle.scale_for_fragment(fragment_id=0, t=2, t_planned=3) == 1.5
    assert oracle.scale_for_fragment(fragment_id=1, t=2, t_planned=2) == 0.75
    with pytest.raises(ValueError, match="planned horizon"):
        oracle.scale_for_fragment(fragment_id=1, t=1, t_planned=3)


def test_json_interface_is_strict_and_loadable(tmp_path):
    payload = {
        "schema": "yeto.outer_lr_drift_surface.v1",
        "output": "log2_lr_multiplier",
        "features": [],
        "terms": [{"coefficient": 0.0, "powers": {}}],
        "drift_scale_bounds": {"min": 0.5, "max": 2.0},
    }
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(payload))
    assert FactorialSurface.from_json(path).drift_scale(
        mu=0.9, t=1, t_planned=1
    ) == 1.0

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        FactorialSurface.from_mapping(payload)


def test_round_robin_planned_horizon_matches_syncer_schedule():
    assert [
        planned_fragment_steps(total_steps=10, fragments=4, fragment_id=fid)
        for fid in range(4)
    ] == [3, 3, 2, 2]
    assert planned_fragment_steps(total_steps=2, fragments=4, fragment_id=2) == 0


def test_invalid_age_and_missing_active_spectral_feature_fail_closed():
    with pytest.raises(ValueError, match="1 <= t <= T_planned"):
        outer_lr_scale("transient", mu=0.9, t=0, t_planned=4)
    with pytest.raises(ValueError, match="spectral feature 'window'"):
        outer_lr_scale(
            "measured-drift",
            mu=0.9,
            t=1,
            t_planned=5,
            surface=_surface(coefficients=(0.0, 0.0, 0.1)),
        )
