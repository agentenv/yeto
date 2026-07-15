#!/usr/bin/env python3
"""Deterministic colored-input quadratic falsification suite.

This CPU-only suite separates facts that a scalar lag-1 correlation cannot:
higher-lag spectrum, finite buffer age, nonstationarity, and the curvature
orientation of equal-norm transverse steps.  It emits JSON so the results can
be pinned in preregistration/CI rather than interpreted from a notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


Vector = list[float]
Series = list[Vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(left, right, strict=True)]


def scale(value: Sequence[float], factor: float) -> Vector:
    return [factor * item for item in value]


def diagonal_matvec(diagonal: Sequence[float], value: Sequence[float]) -> Vector:
    return [entry * item for entry, item in zip(diagonal, value, strict=True)]


def quadratic_loss(diagonal: Sequence[float], value: Sequence[float]) -> float:
    return 0.5 * dot(value, diagonal_matvec(diagonal, value))


def _standardize(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    centered = [value - mean for value in values]
    variance = statistics.fmean(value * value for value in centered)
    if variance <= 0.0:
        raise ValueError("cannot standardize a zero-variance process")
    inverse_sd = 1.0 / math.sqrt(variance)
    return [value * inverse_sd for value in centered]


def _ar2_series(
    length: int,
    rng: random.Random,
    *,
    a1: float,
    a2: float,
    innovation_sd: float = 1.0,
    burnin: int = 512,
) -> list[float]:
    values = [0.0, 0.0]
    for _ in range(length + burnin):
        values.append(
            a1 * values[-1]
            + a2 * values[-2]
            + innovation_sd * rng.gauss(0.0, 1.0)
        )
    return _standardize(values[-length:])


def _scalar_process(kind: str, length: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    burnin = 512
    if kind == "ar1":
        phi = 0.72
        values = [0.0]
        innovation_sd = math.sqrt(1.0 - phi * phi)
        for _ in range(length + burnin):
            values.append(phi * values[-1] + innovation_sd * rng.gauss(0.0, 1.0))
        return _standardize(values[-length:])
    if kind == "ar2":
        return _ar2_series(length, rng, a1=0.65, a2=-0.25, burnin=burnin)
    if kind == "ma":
        innovations = [rng.gauss(0.0, 1.0) for _ in range(length + burnin + 3)]
        values = [
            innovations[index]
            + 0.8 * innovations[index - 1]
            - 0.3 * innovations[index - 2]
            for index in range(2, len(innovations))
        ]
        return _standardize(values[-length:])
    if kind == "oscillatory":
        radius, omega = 0.94, 0.62
        return _ar2_series(
            length,
            rng,
            a1=2.0 * radius * math.cos(omega),
            a2=-(radius**2),
            burnin=burnin,
        )
    if kind == "nonstationary":
        values: list[float] = []
        state = 0.0
        for index in range(length):
            first_half = index < length // 2
            phi = 0.84 if first_half else -0.45
            amplitude = 0.55 + index / max(1, length - 1)
            state = phi * state + math.sqrt(1.0 - phi * phi) * rng.gauss(0.0, 1.0)
            values.append(amplitude * state)
        return _standardize(values)
    raise ValueError(f"unknown process kind: {kind}")


def generate_process(kind: str, steps: int, dimension: int, seed: int) -> Series:
    if steps < 16:
        raise ValueError("steps must be at least 16")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    columns = [
        _scalar_process(kind, steps, seed + 104729 * axis)
        for axis in range(dimension)
    ]
    return [[columns[axis][step] for axis in range(dimension)] for step in range(steps)]


def empirical_kernel(series: Sequence[Sequence[float]], max_lag: int) -> list[float]:
    if not series:
        raise ValueError("series cannot be empty")
    max_lag = min(max_lag, len(series) - 1)
    output = [1.0]
    for lag in range(1, max_lag + 1):
        current = series[lag:]
        past = series[:-lag]
        numerator = math.fsum(dot(left, right) for left, right in zip(current, past, strict=True))
        current_energy = math.fsum(dot(value, value) for value in current)
        past_energy = math.fsum(dot(value, value) for value in past)
        denominator = math.sqrt(current_energy * past_energy)
        output.append(numerator / denominator)
    return output


def nesterov_filter(series: Sequence[Sequence[float]], mu: float) -> dict[str, Any]:
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must lie in [0, 1)")
    dimension = len(series[0])
    buffer = [0.0] * dimension
    output: Series = []
    direct_error = 0.0
    input_energy = 0.0
    output_energy = 0.0
    alignment = 0.0
    for time, delta in enumerate(series):
        previous_buffer = buffer
        direction = add(scale(delta, 1.0 + mu), scale(previous_buffer, mu * mu))
        buffer = add(scale(previous_buffer, mu), delta)
        output.append(direction)

        direct = scale(delta, 1.0 + mu)
        for lag in range(1, time + 1):
            direct = add(direct, scale(series[time - lag], mu ** (lag + 1)))
        direct_error = max(
            direct_error,
            max(abs(left - right) for left, right in zip(direction, direct, strict=True)),
        )
        input_energy += dot(delta, delta)
        output_energy += dot(direction, direction)
        alignment += dot(direction, delta)
    return {
        "aligned_ratio": alignment / input_energy,
        "energy_amplification": output_energy / input_energy,
        "max_direct_convolution_error": direct_error,
        "output": output,
    }


def simulate_quadratic(
    noise: Series,
    *,
    mu: float,
    eta: float,
    curvature: Sequence[float],
    noise_scale: float,
) -> dict[str, float]:
    dimension = len(curvature)
    if any(len(value) != dimension for value in noise):
        raise ValueError("noise dimension does not match curvature")
    state = [1.0 / (axis + 1.0) for axis in range(dimension)]
    buffer = [0.0] * dimension
    losses = [quadratic_loss(curvature, state)]
    deltas: Series = []
    directions: Series = []
    for innovation in noise:
        gradient = diagonal_matvec(curvature, state)
        delta = add(gradient, scale(innovation, noise_scale))
        previous_buffer = buffer
        direction = add(scale(delta, 1.0 + mu), scale(previous_buffer, mu * mu))
        buffer = add(scale(previous_buffer, mu), delta)
        state = add(state, scale(direction, -eta))
        loss = quadratic_loss(curvature, state)
        if not math.isfinite(loss):
            raise FloatingPointError("quadratic simulation diverged to a non-finite loss")
        deltas.append(delta)
        directions.append(direction)
        losses.append(loss)
    tail = losses[max(1, len(losses) * 3 // 4) :]
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_mean_loss": statistics.fmean(tail),
        "max_loss": max(losses),
        "delta_rho1": empirical_kernel(deltas, 1)[1],
        "direction_energy_ratio": math.fsum(dot(value, value) for value in directions)
        / math.fsum(dot(value, value) for value in deltas),
    }


def ar2_kernel(a1: float, a2: float, max_lag: int) -> list[float]:
    rho1 = a1 / (1.0 - a2)
    kernel = [1.0, rho1]
    for _ in range(2, max_lag + 1):
        kernel.append(a1 * kernel[-1] + a2 * kernel[-2])
    return kernel[: max_lag + 1]


def finite_filter_moments(mu: float, kernel: Sequence[float], age: int) -> dict[str, float]:
    if age <= 0:
        raise ValueError("age must be positive")
    if len(kernel) < age:
        raise ValueError("kernel must include all lags through age-1")
    coefficients = [1.0 + mu] + [mu ** (lag + 1) for lag in range(1, age)]
    aligned = math.fsum(coefficients[lag] * kernel[lag] for lag in range(age))
    energy = math.fsum(
        coefficients[left] * coefficients[right] * kernel[abs(left - right)]
        for left in range(age)
        for right in range(age)
    )
    return {"aligned_ratio": aligned, "energy_amplification": energy}


def same_rho1_contrast(mu: float = 0.9, age: int = 64) -> dict[str, Any]:
    target_rho1 = 0.4
    ar1_kernel = [target_rho1**lag for lag in range(age)]
    # Yule-Walker rho_1 = a1 / (1-a2) = .4, but rho_2 = -.176.
    a1, a2 = 0.56, -0.4
    ar2_values = ar2_kernel(a1, a2, age - 1)
    ar1_moments = finite_filter_moments(mu, ar1_kernel, age)
    ar2_moments = finite_filter_moments(mu, ar2_values, age)
    return {
        "target_rho1": target_rho1,
        "age": age,
        "ar1": {
            "kernel_head": ar1_kernel[:8],
            "rho2": ar1_kernel[2],
            **ar1_moments,
        },
        "ar2": {
            "a1": a1,
            "a2": a2,
            "kernel_head": ar2_values[:8],
            "rho2": ar2_values[2],
            **ar2_moments,
        },
        "aligned_difference_ar1_minus_ar2": ar1_moments["aligned_ratio"]
        - ar2_moments["aligned_ratio"],
        "energy_difference_ar1_minus_ar2": ar1_moments["energy_amplification"]
        - ar2_moments["energy_amplification"],
        "conclusion": "same_rho1_is_not_sufficient_for_filter_moments",
    }


def finite_buffer_age_sweep(mu: float = 0.9, phi: float = 0.6) -> list[dict[str, float]]:
    ages = (1, 2, 5, 10, 20, 50, 100, 320)
    kernel = [phi**lag for lag in range(max(ages))]
    return [
        {"age": age, **finite_filter_moments(mu, kernel, age)}
        for age in ages
    ]


def curvature_rotation(eta: float = 0.02) -> dict[str, Any]:
    curvature = [1.0, 10.0, 100.0]
    state = [1.0, 0.0, 0.0]
    gradient = diagonal_matvec(curvature, state)
    baseline_direction = gradient
    flat_transverse = [0.0, 1.0, 0.0]
    sharp_transverse = [0.0, 0.0, 1.0]
    baseline_next = add(state, scale(baseline_direction, -eta))
    baseline_loss = quadratic_loss(curvature, baseline_next)

    def arm(name: str, transverse: Vector) -> dict[str, float | str]:
        direction = add(baseline_direction, transverse)
        next_state = add(state, scale(direction, -eta))
        observed_excess = quadratic_loss(curvature, next_state) - baseline_loss
        predicted_excess = 0.5 * eta * eta * dot(
            transverse, diagonal_matvec(curvature, transverse)
        )
        return {
            "name": name,
            "transverse_norm": math.sqrt(dot(transverse, transverse)),
            "gradient_dot_transverse": dot(gradient, transverse),
            "curvature_energy": dot(transverse, diagonal_matvec(curvature, transverse)),
            "observed_excess_loss": observed_excess,
            "predicted_excess_loss": predicted_excess,
            "prediction_error": observed_excess - predicted_excess,
        }

    return {
        "eta": eta,
        "curvature": curvature,
        "baseline_loss_after_step": baseline_loss,
        "arms": [arm("flat", flat_transverse), arm("sharp", sharp_transverse)],
        "conclusion": "equal_euclidean_transverse_norm_has_curvature_dependent_loss",
    }


def _theoretical_kernel(kind: str, max_lag: int) -> list[float] | None:
    if kind == "ar1":
        return [0.72**lag for lag in range(max_lag + 1)]
    if kind == "ar2":
        return ar2_kernel(0.65, -0.25, max_lag)
    if kind == "ma":
        coefficients = [1.0, 0.8, -0.3]
        gamma0 = math.fsum(value * value for value in coefficients)
        kernel = [1.0]
        for lag in range(1, max_lag + 1):
            covariance = math.fsum(
                coefficients[index] * coefficients[index + lag]
                for index in range(len(coefficients) - lag)
            ) if lag < len(coefficients) else 0.0
            kernel.append(covariance / gamma0)
        return kernel
    if kind == "oscillatory":
        radius, omega = 0.94, 0.62
        return ar2_kernel(2.0 * radius * math.cos(omega), -(radius**2), max_lag)
    return None


def run_suite(
    *,
    seed: int = 20260714,
    steps: int = 2048,
    dimension: int = 4,
    mu: float = 0.9,
    eta: float = 0.02,
) -> dict[str, Any]:
    curvature = [0.5 + 1.5 * axis for axis in range(dimension)]
    process_rows: list[dict[str, Any]] = []
    for offset, kind in enumerate(("ar1", "ar2", "ma", "oscillatory", "nonstationary")):
        series = generate_process(kind, steps, dimension, seed + offset * 1_000_003)
        filtered = nesterov_filter(series, mu)
        row: dict[str, Any] = {
            "kind": kind,
            "empirical_kernel": empirical_kernel(series, 8),
            "theoretical_kernel": _theoretical_kernel(kind, 8),
            "filter": {
                key: value
                for key, value in filtered.items()
                if key != "output"
            },
            "quadratic": {
                "memoryless": simulate_quadratic(
                    series,
                    mu=0.0,
                    eta=eta,
                    curvature=curvature,
                    noise_scale=0.08,
                ),
                "nesterov": simulate_quadratic(
                    series,
                    mu=mu,
                    eta=eta,
                    curvature=curvature,
                    noise_scale=0.08,
                ),
            },
        }
        if kind == "nonstationary":
            midpoint = len(series) // 2
            row["first_half_kernel"] = empirical_kernel(series[:midpoint], 4)
            row["second_half_kernel"] = empirical_kernel(series[midpoint:], 4)
        process_rows.append(row)
    return {
        "schema_version": "1.0",
        "suite": "colored_input_quadratic_falsification",
        "deterministic_config": {
            "seed": seed,
            "steps": steps,
            "dimension": dimension,
            "mu": mu,
            "eta": eta,
            "curvature_diagonal": curvature,
        },
        "processes": process_rows,
        "same_rho1_different_spectrum": same_rho1_contrast(mu=mu),
        "finite_buffer_age": finite_buffer_age_sweep(mu=mu),
        "curvature_rotation": curvature_rotation(eta=eta),
        "interpretation_scope": [
            "deterministic_cpu_falsification_suite",
            "not_a_neural_training_replication",
            "rho1_only_is_tested_against_full_temporal_geometry",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_suite(
            seed=args.seed,
            steps=args.steps,
            dimension=args.dimension,
            mu=args.mu,
            eta=args.eta,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (ValueError, FloatingPointError) as exc:
        print(f"COLORED_QUADRATIC_ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
