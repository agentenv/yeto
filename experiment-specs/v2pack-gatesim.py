#!/usr/bin/env python3
"""Deterministic, CPU-only feasibility simulations for registrations v14--v17.

The simulations test whether the frozen decision rules are evaluable and powered
under declared prospective alternatives.  They are not evidence that any of
those alternatives is scientifically true.  No launcher, result directory, or
node-facing artifact is created by this program.

The committed reports are reproduced with::

    python3 experiment-specs/v2pack-gatesim.py --verify

Use ``--emit v14`` (or v15/v16/v17/all) to print canonical JSON to stdout.
The program intentionally has no write mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
REPLICATES = 20_000
MANDATORY_P_EVAL = 0.80
T_CRITICAL_95_DF4 = 2.7764451051977987
T_CRITICAL_95_DF6 = 2.4469118487916806

REPORT_PATHS = {
    "v14": ROOT / "outer-mup-v14-transfer-matrix-gatesim.json",
    "v15": ROOT / "outer-mup-v15-multiseed-scale-panel-gatesim.json",
    "v16": ROOT / "outer-mup-v16-pythia-redesign-gatesim.json",
    "v17": ROOT / "outer-mup-v17-reproduce-overturn-gatesim.json",
}

SOURCE_SHA256 = {
    "g4c_readout": "16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa",
    "g9_joint_readout": "4d42a5f133684822f0fc81c69c3348610c7b3824836b1ecf937abb6e9515bcaf",
    "g10_readout": "46adf7b8b11a9434632e3470248b3adc3b6bb9e6a71d1aa5b4df59e84390386f",
    "v13b_readout": "868d8aaa3c422bdb4475302d2071ca0082108e813aab70c5f9061ef90f156f48",
    "snoo_primary_source_text": "b8a1fce9c0ddcb8debdc9c9bc3e714ac60d0c93895922cc5d44b6d6a35ef4aec",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def probability(count: int, total: int) -> float:
    return count / total


def mean_ci_t(
    values: list[float], critical_value: float = T_CRITICAL_95_DF4
) -> tuple[float, float]:
    """Two-sided 95% Student interval used by v14 and v17."""

    if len(values) < 2:
        raise ValueError("registered Student interval requires at least two values")
    center = statistics.mean(values)
    half = critical_value * statistics.stdev(values) / math.sqrt(len(values))
    return center - half, center + half


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def normal_profile(rng: random.Random, n: int, sd: float) -> list[float]:
    return [rng.gauss(0.0, sd) for _ in range(n)]


# ---------------------------------------------------------------------------
# v14: directed transfer matrix

V14_T = (2, 5, 20, 40)
V14_SEEDS = (1401, 1409, 1423, 1427, 1429)
V14_CONTEXTS = {
    "FIXED_H512": {t: {"H": 512, "S": 512 * t} for t in V14_T},
    "FIXED_S2560": {t: {"H": 2560 // t, "S": 2560} for t in V14_T},
}
V14_FIXED_H_RATES = {
    2: 0.007821882581822885,
    5: 0.003191644884294105,
    20: 0.0008223020084526104,
    40: 0.0003971228256207733,
}
V14_CONTEXT_RATIOS = {
    2: 0.9385818429796292,
    5: 1.0,
    20: 1.100646942793633,
    40: 1.154707661608376,
}
V14_RATES = {
    "FIXED_H512": dict(V14_FIXED_H_RATES),
    "FIXED_S2560": {
        t: V14_FIXED_H_RATES[t] * V14_CONTEXT_RATIOS[t] for t in V14_T
    },
}
V14_PRACTICAL_MARGIN = 0.10
V14_WINSOR_LIMIT = 0.75

# Paired penalty profiles, in bits/token, retained from G10.  The 9.5048
# scientific-divergence value is not deleted: the registered inferential
# estimand clips all seed-level penalties symmetrically at +/-0.75 and reports
# the raw mean separately.  Centering the bounded values yields these profiles.
_G10_UP_VOLATILE = [0.610896465378124, 0.75, 0.5073930155229649]
_G10_UP_STABLE = [0.48514097420302094, 0.5161231881603399, 0.5590176328843328]
_G10_DOWN = [0.06988370781729199, 0.07439199362427062, 0.08622200259696124]


def centered(values: Iterable[float]) -> list[float]:
    values = list(values)
    center = statistics.mean(values)
    return [value - center for value in values]


V14_NOISE_PROFILES = {
    "up_volatile_bounded": centered(_G10_UP_VOLATILE),
    "up_stable": centered(_G10_UP_STABLE),
    "down": centered(_G10_DOWN),
    # The pooled G4C paired-contrast SD was 0.068177 bits and the maximum was
    # 0.163605 bits.  This deterministic five-point stress profile has the
    # maximum measured SD and is mixed into every pair's power calculation.
    "g4c_max_sd_stress": [
        -0.21814057547279987,
        -0.10907028773639993,
        0.0,
        0.10907028773639993,
        0.21814057547279987,
    ],
}


def v14_expected_penalty(source_t: int, target_t: int) -> float:
    octaves = abs(math.log2(target_t / source_t))
    if source_t < target_t:
        # The 0.31-bit one-octave floor is below both stable G10 upward
        # measurements (+0.432 in the independent audit and +0.520 in the
        # fresh T5->T40 pair).  Longer transfers rise by 0.08 bit/octave and
        # are conservatively capped at the stable fresh +0.520 result.
        return min(0.52, 0.31 + 0.08 * max(0.0, octaves - 1.0))
    return min(0.075, 0.025 * octaves)


def v14_pair_label(values: list[float]) -> str:
    low, high = mean_ci_t(values)
    if low > V14_PRACTICAL_MARGIN:
        return "PAIR_PENALTY"
    if high < -V14_PRACTICAL_MARGIN:
        return "PAIR_BENEFIT"
    return "PAIR_NO_DECISIVE_PENALTY"


def simulate_v14() -> dict[str, Any]:
    rng = random.Random(20260814)
    pairs = [
        (context, source_t, target_t)
        for context in V14_CONTEXTS
        for source_t in V14_T
        for target_t in V14_T
        if source_t != target_t
    ]
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    asymmetry = Counter()
    evaluable = 0
    for _ in range(REPLICATES):
        # Shared banked indices preserve cross-pair common seed effects.
        banked_indices = [rng.randrange(15) for _ in V14_SEEDS]
        seed_values: dict[tuple[str, int, int], list[float]] = {}
        for context, source_t, target_t in pairs:
            upward = source_t < target_t
            expected = v14_expected_penalty(source_t, target_t)
            profile_names = (
                ("up_stable", "up_volatile_bounded", "g4c_max_sd_stress")
                if upward
                else ("down", "down", "g4c_max_sd_stress")
            )
            values = []
            for banked_index in banked_indices:
                profile_name = profile_names[banked_index % len(profile_names)]
                profile = V14_NOISE_PROFILES[profile_name]
                residual = profile[banked_index % len(profile)]
                # Fixed-S is conservatively stressed by 10% because it is a
                # transported rather than directly measured rate panel.
                if context == "FIXED_S2560":
                    residual *= 1.10
                value = max(-V14_WINSOR_LIMIT, min(V14_WINSOR_LIMIT, expected + residual))
                values.append(value)
            seed_values[(context, source_t, target_t)] = values
            pair_id = f"{context}:T{source_t}_TO_T{target_t}"
            labels[pair_id][v14_pair_label(values)] += 1

        upward_by_seed = []
        downward_by_seed = []
        for seed_index in range(len(V14_SEEDS)):
            upward_by_seed.append(
                statistics.mean(
                    values[seed_index]
                    for (context, source_t, target_t), values in seed_values.items()
                    if source_t < target_t
                )
            )
            downward_by_seed.append(
                statistics.mean(
                    values[seed_index]
                    for (context, source_t, target_t), values in seed_values.items()
                    if source_t > target_t
                )
            )
        difference = [u - d for u, d in zip(upward_by_seed, downward_by_seed)]
        diff_low, diff_high = mean_ci_t(difference)
        up_low, _ = mean_ci_t(upward_by_seed)
        down_point = statistics.mean(downward_by_seed)
        if (
            diff_low > V14_PRACTICAL_MARGIN
            and up_low > V14_PRACTICAL_MARGIN
            and abs(down_point) <= V14_PRACTICAL_MARGIN
        ):
            asymmetry["ASYMMETRY_CONFIRMED"] += 1
        elif diff_high < -V14_PRACTICAL_MARGIN:
            asymmetry["ASYMMETRY_REVERSED"] += 1
        else:
            asymmetry["ASYMMETRY_NULL"] += 1
        evaluable += 1

    per_pair = {}
    for context, source_t, target_t in pairs:
        pair_id = f"{context}:T{source_t}_TO_T{target_t}"
        expected_label = (
            "PAIR_PENALTY" if source_t < target_t else "PAIR_NO_DECISIVE_PENALTY"
        )
        per_pair[pair_id] = {
            "context": context,
            "source_T": source_t,
            "target_T": target_t,
            "source_eta_exact": V14_RATES[context][source_t],
            "target_comparator_eta_exact": V14_RATES[context][target_t],
            "prospective_alternative_penalty_bits": v14_expected_penalty(source_t, target_t),
            "expected_label": expected_label,
            "P_expected_label": probability(labels[pair_id][expected_label], REPLICATES),
            "label_probabilities": {
                label: probability(labels[pair_id][label], REPLICATES)
                for label in (
                    "PAIR_PENALTY",
                    "PAIR_NO_DECISIVE_PENALTY",
                    "PAIR_BENEFIT",
                )
            },
        }

    return {
        "schema": "yeto_outer_mup_v14_transfer_matrix_gatesim_v1",
        "status": "PASS",
        "pre_outcome": True,
        "scope": "gate evaluability and power under the declared G10-derived asymmetry alternative; not scientific evidence for transfer penalties",
        "sources": {
            "g4c_readout_sha256": SOURCE_SHA256["g4c_readout"],
            "g10_readout_sha256": SOURCE_SHA256["g10_readout"],
            "g4c_pooled_paired_contrast_sd_bits": 0.06817707523738205,
            "g4c_maximum_paired_contrast_sd_bits": 0.1636054316045999,
            "g10_raw_scientific_divergence_retained_bits": 9.504813567101717,
        },
        "simulation": {
            "replicates": REPLICATES,
            "rng_seed": 20260814,
            "fresh_seeds": list(V14_SEEDS),
            "P_evaluable": probability(evaluable, REPLICATES),
            "mandatory_P_evaluable": MANDATORY_P_EVAL,
            "clears_evaluability_bar": probability(evaluable, REPLICATES) >= MANDATORY_P_EVAL,
        },
        "estimator": {
            "seed_level_penalty": "raw transfer-minus-fresh-comparator endpoint NLL in bits/token, symmetrically clipped to [-0.75,+0.75] only for the bounded primary estimand",
            "raw_values": "all unclipped seed values, including scientific divergence, are mandatory descriptive outputs",
            "interval": "two-sided 95% paired Student t interval across five fresh seeds",
            "practical_margin_bits": V14_PRACTICAL_MARGIN,
        },
        "per_pair": per_pair,
        "aggregate_asymmetry": {
            "closed_vocabulary": [
                "ASYMMETRY_CONFIRMED",
                "ASYMMETRY_NULL",
                "ASYMMETRY_REVERSED",
            ],
            "P_ASYMMETRY_CONFIRMED": probability(
                asymmetry["ASYMMETRY_CONFIRMED"], REPLICATES
            ),
            "P_ASYMMETRY_NULL": probability(asymmetry["ASYMMETRY_NULL"], REPLICATES),
            "P_ASYMMETRY_REVERSED": probability(
                asymmetry["ASYMMETRY_REVERSED"], REPLICATES
            ),
            "alternative": "upward source_T<target_T penalties have a conservative 0.31-bit one-octave floor, rise by 0.08 bit per additional log2 horizon octave, and cap at the stable fresh G10 +0.52 result; downward penalties scale as 0.025 bits per octave capped at 0.075",
        },
    }


# ---------------------------------------------------------------------------
# v15: multi-seed 7B and 27B-LoRA panel

V15_SEEDS = (1501, 1511, 1523)
V15_T_7B = (2, 5, 20)
V15_T_27B = (5, 20)
V15_GRID_OFFSETS = (-1.25, -0.75, -0.25, 0.25, 0.75, 1.25)
V15_ABSOLUTE_BAND = 0.21
V15_RELATIVE_BAND = 0.16
V15_RELATIVE_RESIDUAL = -0.051907000000000036


def drift_bits(parameters_b: float) -> float:
    slope = (-0.50 - (-0.40)) / math.log2(7.0 / 1.7)
    return -0.50 + slope * math.log2(parameters_b / 7.0)


def raw_surface_D(t: int) -> float:
    gamma = 1.3489008177233357
    alpha = -0.9098513603667141
    beta = -0.10723867757601385
    epsilon = 0.16020840966569636
    s = 512 * t
    u = (t - 5) / 5
    v = math.log2(s / 5120)
    return 2 ** (gamma + alpha * u + beta * v + epsilon * u * u)


def v15_centers() -> dict[str, dict[str, dict[str, float]]]:
    mu0_t5 = 0.007827013
    raw_t5 = 0.002071630
    slope = -0.756529843311295
    seven = {}
    for t in V15_T_7B:
        mu0 = mu0_t5 * (t / 5) ** slope
        raw = mu0 * (raw_t5 / mu0_t5) * (raw_surface_D(t) / raw_surface_D(5))
        seven[f"T{t}"] = {"mu0": mu0, "raw_mu0p9": raw}
    lora_mu0_t5 = 0.28
    lora_mu0_t20 = lora_mu0_t5 * 0.35036736670682456
    twenty_seven = {
        "T5": {
            "mu0": lora_mu0_t5,
            "raw_mu0p9": lora_mu0_t5 * 0.1 * 1.7416157949788522,
        },
        "T20": {
            "mu0": lora_mu0_t20,
            "raw_mu0p9": lora_mu0_t20 * 0.1 * 1.2806943474449415,
        },
    }
    return {"qwen2p5_7b_full": seven, "qwen3p6_27b_lora": twenty_seven}


def simulate_v15() -> dict[str, Any]:
    rng = random.Random(20260815)
    counts = Counter()
    for _ in range(REPLICATES):
        errors_7b: dict[tuple[int, str], float] = {}
        for t in V15_T_7B:
            common = rng.gauss(0.0, 0.10 / math.sqrt(len(V15_SEEDS)))
            mu0_specific = rng.gauss(0.0, 0.0632194750783273 / math.sqrt(len(V15_SEEDS)))
            raw_specific = rng.gauss(0.0, 0.0632194750783273 / math.sqrt(len(V15_SEEDS)))
            errors_7b[(t, "mu0")] = common + mu0_specific
            errors_7b[(t, "raw_mu0p9")] = (
                common + raw_specific + V15_RELATIVE_RESIDUAL
            )
        absolute_hits_7b = sum(abs(value) <= V15_ABSOLUTE_BAND for value in errors_7b.values())
        relative_hits_7b = sum(
            abs(errors_7b[(t, "raw_mu0p9")] - errors_7b[(t, "mu0")])
            <= V15_RELATIVE_BAND
            for t in V15_T_7B
        )
        absolute_pass_7b = absolute_hits_7b >= 5
        relative_pass_7b = relative_hits_7b >= 2
        counts["7b_absolute_pass"] += absolute_pass_7b
        counts["7b_relative_pass"] += relative_pass_7b
        counts["7b_both_pass"] += absolute_pass_7b and relative_pass_7b
        # Count the important separation: the relative gate can pass even if
        # the absolute common-mode gate does not.
        counts["7b_relative_only"] += relative_pass_7b and not absolute_pass_7b

        errors_27b: dict[tuple[int, str], float] = {}
        lora_sd = 1.5 * 0.0632194750783273
        for t in V15_T_27B:
            common = rng.gauss(0.0, 0.12 / math.sqrt(len(V15_SEEDS)))
            mu0_specific = rng.gauss(0.0, lora_sd / math.sqrt(len(V15_SEEDS)))
            raw_specific = rng.gauss(0.0, lora_sd / math.sqrt(len(V15_SEEDS)))
            errors_27b[(t, "mu0")] = common + mu0_specific
            errors_27b[(t, "raw_mu0p9")] = (
                common + raw_specific + V15_RELATIVE_RESIDUAL
            )
        absolute_hits_27b = sum(abs(value) <= V15_ABSOLUTE_BAND for value in errors_27b.values())
        relative_hits_27b = sum(
            abs(errors_27b[(t, "raw_mu0p9")] - errors_27b[(t, "mu0")])
            <= V15_RELATIVE_BAND
            for t in V15_T_27B
        )
        counts["27b_absolute_pass"] += absolute_hits_27b >= 3
        counts["27b_relative_pass"] += relative_hits_27b >= 1

    centers = v15_centers()
    grids = {}
    for model, horizons in centers.items():
        grids[model] = {}
        for horizon, arms in horizons.items():
            grids[model][horizon] = {}
            for arm, center_value in arms.items():
                grids[model][horizon][arm] = {
                    "center": center_value,
                    "offsets_log2": list(V15_GRID_OFFSETS),
                    "etas": [center_value * 2**offset for offset in V15_GRID_OFFSETS],
                }
    return {
        "schema": "yeto_outer_mup_v15_multiseed_scale_panel_gatesim_v1",
        "status": "PASS",
        "pre_outcome": True,
        "scope": "bracketing and separate absolute/arm-relative gate feasibility; not evidence that scale transport is true",
        "sources": {
            "g4c_readout_sha256": SOURCE_SHA256["g4c_readout"],
            "g9_joint_readout_sha256": SOURCE_SHA256["g9_joint_readout"],
            "observed_common_mode_drift_bits": {"1p7B": -0.40, "7B": -0.50},
            "observed_7b_raw_minus_mu0_residual_bits": V15_RELATIVE_RESIDUAL,
            "v7_pooled_residual_population_sd": 0.0632194750783273,
            "v7_maximum_coordinate_seed_sd": 0.25699220611018175,
        },
        "band_derivation": {
            "drift_slope_bits_per_log2_parameter_octave": (-0.10) / math.log2(7.0 / 1.7),
            "drift_correction_bits": {
                "qwen2p5_7b_full": drift_bits(7.0),
                "qwen3p6_27b_lora": drift_bits(27.0),
            },
            "absolute_half_width_bits": V15_ABSOLUTE_BAND,
            "absolute_rule": "round outward to 0.21 from 4*abs(measured 7B raw-minus-mu0 residual)=0.207628 bits",
            "arm_relative_half_width_bits": V15_RELATIVE_BAND,
            "relative_rule": "round outward to 0.16 from 3*abs(measured 7B raw-minus-mu0 residual)=0.155721 bits",
        },
        "registered_grids": grids,
        "simulation": {
            "replicates": REPLICATES,
            "rng_seed": 20260815,
            "seeds_per_arm": len(V15_SEEDS),
            "seeds": list(V15_SEEDS),
            "P_evaluable": 1.0,
            "mandatory_P_evaluable": MANDATORY_P_EVAL,
            "P_7B_ABSOLUTE_TRANSPORT_PASS": probability(counts["7b_absolute_pass"], REPLICATES),
            "P_7B_ARM_RELATIVE_TRANSPORT_PASS": probability(counts["7b_relative_pass"], REPLICATES),
            "P_7B_both_pass": probability(counts["7b_both_pass"], REPLICATES),
            "P_7B_relative_pass_while_absolute_fails": probability(counts["7b_relative_only"], REPLICATES),
            "P_27B_ABSOLUTE_TRANSPORT_PASS": probability(counts["27b_absolute_pass"], REPLICATES),
            "P_27B_ARM_RELATIVE_TRANSPORT_PASS": probability(counts["27b_relative_pass"], REPLICATES),
        },
        "noise_model": {
            "7b": "per horizon, shared common-mode N(0,0.10^2/n) plus independent arm N(0,0.063219^2/n); raw arm shifted by measured -0.051907-bit residual",
            "27b_lora": "per horizon, shared common-mode N(0,0.12^2/n) plus independent arm N(0,(1.5*0.063219)^2/n); raw arm shifted by measured -0.051907-bit residual",
            "common_mode_is_shared_before_arm_subtraction": True,
        },
    }


# ---------------------------------------------------------------------------
# v16: Pythia-160M redesign

V16_SEEDS = (
    1601,
    1607,
    1609,
    1613,
    1619,
    1621,
    1627,
    1637,
    1657,
    1663,
    1667,
    1669,
    1693,
    1697,
    1699,
    1709,
    1721,
)
V16_OFFSETS = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
V16_CENTERS = {
    "T2_mu0": 0.007959003015874582,
    "T2_mu09": 0.0034223165582977205,
    "T5_mu0": 0.005427397642553673 / math.sqrt(2.0),
    "T5_mu09": 0.0014891323817427511 / math.sqrt(2.0),
    "T20_mu0": 0.1754097492953639 * math.sqrt(2.0),
    "T20_mu09": 0.0004249680901238146,
}
V16_CORE_SD = {
    "T2_mu0": 0.113923,
    "T2_mu09": 0.165455,
    "T5_mu0": 0.114362,
    "T5_mu09": 0.039258,
    "T20_mu0": 0.145010,
    "T20_mu09": 0.439958,
}
V16_DIVERGENCE = {
    "T20_mu0": {"rung_index": 2, "spike_nll": 129.75700971048354},
    "T20_mu09": {"rung_index": 5, "spike_nll": 39.37726015831847},
}
V16_BOOTSTRAP_MINIMUM = 7000


def solve_three_by_three(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-15:
            raise ValueError("singular quadratic design")
        for index in range(column, 4):
            augmented[column][index] /= divisor
        for row in range(3):
            if row == column:
                continue
            multiplier = augmented[row][column]
            for index in range(column, 4):
                augmented[row][index] -= multiplier * augmented[column][index]
    return [augmented[row][3] for row in range(3)]


def quadratic_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    design = [[x * x, x, 1.0] for x in xs]
    gram = [
        [sum(row[i] * row[j] for row in design) for j in range(3)]
        for i in range(3)
    ]
    rhs = [sum(row[i] * y for row, y in zip(design, ys)) for i in range(3)]
    a, b, _ = solve_three_by_three(gram, rhs)
    vertex = -b / (2.0 * a) if a > 0 else math.inf
    return a, vertex


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(
        math.comb(n, index) * p**index * (1.0 - p) ** (n - index)
        for index in range(k + 1)
    )


def simulate_v16() -> dict[str, Any]:
    rng = random.Random(20260816)
    accepted_counts = Counter()
    evaluable = 0
    monotone = 0
    valid_bootstrap_counts: list[int] = []
    for _ in range(REPLICATES):
        # Category 1 is the paired v13b seed carrying both one-sided divergence
        # spikes.  Sampling one category per fresh seed preserves that observed
        # cross-curve dependence.
        categories = [rng.randrange(3) for _ in V16_SEEDS]
        shock_count = sum(category == 1 for category in categories)
        fitted: dict[str, float] = {}
        all_accepted = True
        for curve_id, center_value in V16_CENTERS.items():
            xs = [math.log2(center_value) + offset for offset in V16_OFFSETS]
            per_rung = [[] for _ in V16_OFFSETS]
            for category in categories:
                # A fresh Gaussian core is added around the robust v13b
                # family scale.  The one-sided shocks retain their measured
                # amplitudes and 1/3 seed-profile incidence.
                for rung_index, x in enumerate(xs):
                    value = 3.0 + 0.70 * (x - math.log2(center_value)) ** 2
                    value += rng.gauss(0.0, V16_CORE_SD[curve_id])
                    divergence = V16_DIVERGENCE.get(curve_id)
                    if (
                        category == 1
                        and divergence
                        and rung_index == divergence["rung_index"]
                    ):
                        value += divergence["spike_nll"]
                    per_rung[rung_index].append(value)
            robust_means = [statistics.median(values) for values in per_rung]
            a, vertex = quadratic_fit(xs, robust_means)
            accepted = a > 0 and xs[0] - 0.5 < vertex < xs[-1] + 0.5
            accepted_counts[curve_id] += accepted
            if not accepted:
                all_accepted = False
            fitted[curve_id] = 2**vertex if accepted else math.nan

        # Under the registered one-sided-spike model, a paired seed bootstrap
        # refit is invalid exactly when at least half of its 17 resampled seeds
        # carry the T20/mu0 spike.  This binomial probability is deterministic
        # conditional on the simulated observed shock count.
        p_shock = shock_count / len(V16_SEEDS)
        maximum_valid_shocks = (len(V16_SEEDS) - 1) // 2
        valid_probability = binomial_cdf(
            maximum_valid_shocks, len(V16_SEEDS), p_shock
        )
        valid_bootstrap = int(math.floor(10_000 * valid_probability))
        valid_bootstrap_counts.append(valid_bootstrap)
        is_evaluable = all_accepted and valid_bootstrap >= V16_BOOTSTRAP_MINIMUM
        evaluable += is_evaluable
        if is_evaluable:
            d2 = fitted["T2_mu09"] / (0.1 * fitted["T2_mu0"])
            d5 = fitted["T5_mu09"] / (0.1 * fitted["T5_mu0"])
            d20 = fitted["T20_mu09"] / (0.1 * fitted["T20_mu0"])
            monotone += d2 > d5 > d20

    grids = {
        curve_id: {
            "center": center_value,
            "offsets_log2": list(V16_OFFSETS),
            "etas": [center_value * 2**offset for offset in V16_OFFSETS],
        }
        for curve_id, center_value in V16_CENTERS.items()
    }
    p_eval = probability(evaluable, REPLICATES)
    return {
        "schema": "yeto_outer_mup_v16_pythia_redesign_gatesim_v1",
        "status": "PASS" if p_eval >= MANDATORY_P_EVAL else "FAIL",
        "pre_outcome": True,
        "scope": "bracketing, robust-estimator evaluability, and power under a declared monotone D alternative; not evidence that the Pythia family follows that law",
        "source": {
            "v13b_readout_sha256": SOURCE_SHA256["v13b_readout"],
            "observed_complete_cells": 72,
            "observed_accepted_curves": 3,
            "observed_valid_shared_bootstrap_refits": 0,
            "observed_one_sided_divergence_spikes_nll": {
                key: value["spike_nll"] for key, value in V16_DIVERGENCE.items()
            },
            "robust_core_sd_rule": "1.4826 times each curve's observed v13b 75th percentile absolute residual about its rung median",
        },
        "registered_grids": grids,
        "simulation": {
            "replicates": REPLICATES,
            "rng_seed": 20260816,
            "fresh_seed_count": len(V16_SEEDS),
            "fresh_seeds": list(V16_SEEDS),
            "P_evaluable": p_eval,
            "mandatory_P_evaluable": MANDATORY_P_EVAL,
            "clears_evaluability_bar": p_eval >= MANDATORY_P_EVAL,
            "P_monotone_pass_unconditional": probability(monotone, REPLICATES),
            "P_monotone_pass_given_evaluable": (
                probability(monotone, evaluable) if evaluable else 0.0
            ),
            "per_curve_P_accepted": {
                curve_id: probability(accepted_counts[curve_id], REPLICATES)
                for curve_id in V16_CENTERS
            },
            "valid_bootstrap_refits": {
                "minimum": min(valid_bootstrap_counts),
                "p05": quantile(valid_bootstrap_counts, 0.05),
                "median": statistics.median(valid_bootstrap_counts),
                "minimum_required": V16_BOOTSTRAP_MINIMUM,
            },
        },
        "analysis_model": {
            "rung_estimator": "median endpoint NLL across all 17 registered seeds; no finite scientific-divergence value is dropped",
            "curve_fit": "OLS quadratic in log2(eta) through the six rung medians",
            "acceptance": "positive curvature and unconstrained vertex within 0.5 log2 bits outside the registered endpoint range",
            "bootstrap": "10,000 shared paired seed resamples; refit all rung medians and six quadratics; require at least 7,000 all-six accepted refits",
        },
    }


# ---------------------------------------------------------------------------
# v17: reproduce and overturn SNOO

V17_SEEDS = (17011, 17021, 17027, 17029, 17033, 17041, 17047)
V17_PUBLISHED_GAIN_LOG2 = math.log2(1.28)
V17_REPRO_POINT_FLOOR = math.log2(1.15)
V17_SURVIVAL_MARGIN = math.log2(1.05)
V17_SURVIVAL_ALTERNATIVE = math.log2(1.15)
V17_GAIN_SD = 0.10


def v17_reproduced(values: list[float]) -> bool:
    low, _ = mean_ci_t(values, T_CRITICAL_95_DF6)
    return statistics.mean(values) >= V17_REPRO_POINT_FLOOR and low > 0.0


def v17_survives(values: list[float]) -> bool:
    low, _ = mean_ci_t(values, T_CRITICAL_95_DF6)
    return low > V17_SURVIVAL_MARGIN


def simulate_v17_scenario(rng: random.Random, phase_b_mean: float) -> Counter[str]:
    counts = Counter()
    for _ in range(REPLICATES):
        phase_a = [
            V17_PUBLISHED_GAIN_LOG2 + value
            for value in normal_profile(rng, len(V17_SEEDS), V17_GAIN_SD)
        ]
        phase_b = [
            phase_b_mean + value
            for value in normal_profile(rng, len(V17_SEEDS), V17_GAIN_SD)
        ]
        if not v17_reproduced(phase_a):
            verdict = "GAIN_NOT_REPRODUCED"
        elif v17_survives(phase_b):
            verdict = "GAIN_REPRODUCED_AND_SURVIVES"
        else:
            verdict = "GAIN_REPRODUCED_VANISHES"
        counts[verdict] += 1
    return counts


def simulate_v17() -> dict[str, Any]:
    null_counts = simulate_v17_scenario(random.Random(20260817), 0.0)
    survival_counts = simulate_v17_scenario(
        random.Random(20261817), V17_SURVIVAL_ALTERNATIVE
    )
    vocabulary = (
        "GAIN_REPRODUCED_AND_SURVIVES",
        "GAIN_REPRODUCED_VANISHES",
        "GAIN_NOT_REPRODUCED",
    )
    return {
        "schema": "yeto_outer_mup_v17_reproduce_overturn_gatesim_v1",
        "status": "PASS",
        "pre_outcome": True,
        "scope": "decision-rule power under the published fixed-recipe alternative and either a tuned null or tuned survival alternative; not evidence that SNOO reproduces",
        "source": {
            "primary_source": "Kallusky et al., SNOO: Step-K Nesterov Outer Optimizer, arXiv:2510.15830v1",
            "primary_source_text_sha256": SOURCE_SHA256["snoo_primary_source_text"],
            "published_configuration": {"K": 100, "outer_eta": 0.8, "mu": 0.75},
            "published_step_fraction": 0.78,
            "published_speedup": 1.28,
        },
        "simulation": {
            "replicates_per_scenario": REPLICATES,
            "rng_seeds": {"tuned_null": 20260817, "tuned_survival": 20261817},
            "fresh_confirmation_seeds": list(V17_SEEDS),
            "P_evaluable": 1.0,
            "mandatory_P_evaluable": MANDATORY_P_EVAL,
            "assumed_seed_sd_log2_speedup_bits": V17_GAIN_SD,
            "under_reproduction_plus_tuned_null": {
                verdict: probability(null_counts[verdict], REPLICATES)
                for verdict in vocabulary
            },
            "under_reproduction_plus_1p15x_tuned_survival": {
                verdict: probability(survival_counts[verdict], REPLICATES)
                for verdict in vocabulary
            },
        },
        "decision_model": {
            "phase_A_reproduced": "mean log2 step-speedup >= log2(1.15) and lower endpoint of the paired seven-seed 95% Student interval > 0",
            "phase_B_survives": "lower endpoint of the paired seven-seed 95% Student interval for retuned mu=.75 minus retuned mu=0 log2 step-speedup > log2(1.05)",
            "vanishes_semantics": "failure of the preregistered survival test after Phase A reproduction; not an equivalence claim unless the separately reported CI lies inside the equivalence band",
            "closed_vocabulary": list(vocabulary),
        },
    }


def build_reports() -> dict[str, dict[str, Any]]:
    return {
        "v14": simulate_v14(),
        "v15": simulate_v15(),
        "v16": simulate_v16(),
        "v17": simulate_v17(),
    }


def verify(reports: dict[str, dict[str, Any]]) -> int:
    failures = []
    for name, report in reports.items():
        path = REPORT_PATHS[name]
        expected = canonical_bytes(report)
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        observed = path.read_bytes()
        if observed != expected:
            failures.append(
                f"{name}: byte mismatch expected_sha256={hashlib.sha256(expected).hexdigest()} "
                f"observed_sha256={hashlib.sha256(observed).hexdigest()}"
            )
    if failures:
        for failure in failures:
            print(f"VERIFY FAIL: {failure}")
        return 1
    for name in sorted(reports):
        path = REPORT_PATHS[name]
        print(f"VERIFY PASS: {name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit", choices=("v14", "v15", "v16", "v17", "all"), help="print canonical report JSON"
    )
    parser.add_argument("--verify", action="store_true", help="verify committed report bytes")
    args = parser.parse_args()
    if not args.emit and not args.verify:
        parser.error("one of --emit or --verify is required")
    reports = build_reports()
    if args.emit:
        if args.emit == "all":
            print(json.dumps(reports, indent=2, sort_keys=True))
        else:
            print(canonical_bytes(reports[args.emit]).decode("utf-8"), end="")
    if args.verify:
        return verify(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
