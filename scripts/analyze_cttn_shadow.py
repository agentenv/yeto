#!/usr/bin/env python3
"""Analyze cttn_shadow_v1 predictive alignment and apply its frozen gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence


def predictive_alignment(
    z: Sequence[float], future_g: Sequence[float], r_norm: float
) -> float | None:
    """A_t = z_t dot g_{t+4} / (||r_t|| ||g_{t+4}||)."""
    if len(z) != len(future_g) or not math.isfinite(r_norm) or r_norm <= 0.0:
        return None
    g_norm = math.sqrt(sum(float(value) ** 2 for value in future_g))
    if g_norm <= 0.0 or not math.isfinite(g_norm):
        return None
    value = sum(float(a) * float(b) for a, b in zip(z, future_g)) / (
        r_norm * g_norm
    )
    return value if math.isfinite(value) else None


def shadow_samples(records: Iterable[Mapping]) -> list[dict]:
    return [
        dict(record)
        for record in records
        if "cttn_shadow_sample_step" in record
    ]


def decide(samples: Sequence[Mapping], *, expected_samples: int = 32) -> dict:
    if len(samples) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} resolved shadow samples, got {len(samples)}"
        )
    by_fragment: dict[int, list[float]] = defaultdict(list)
    matrix: list[float] = []
    scalar: list[float] = []
    retention: list[float] = []
    bind_count = 0
    for sample in samples:
        fragment = int(sample["cttn_shadow_fragment"])
        matrix_value = float(sample["cttn_shadow_matrix_alignment"])
        scalar_value = float(sample["cttn_shadow_scalar_alignment"])
        retention_value = float(sample["cttn_shadow_retention"])
        if not all(
            math.isfinite(value)
            for value in (matrix_value, scalar_value, retention_value)
        ):
            raise ValueError("shadow sample contains a non-finite statistic")
        matrix.append(matrix_value)
        scalar.append(scalar_value)
        retention.append(retention_value)
        by_fragment[fragment].append(matrix_value)
        bind_count += bool(sample["cttn_shadow_bind"])

    fragment_means = {
        str(fragment): fmean(values) for fragment, values in sorted(by_fragment.items())
    }
    if len(fragment_means) != 4:
        raise ValueError(f"expected four fragments, got {sorted(by_fragment)}")
    positive_fragments = sum(value > 0.0 for value in fragment_means.values())
    nonpositive_fragments = sum(value <= 0.0 for value in fragment_means.values())
    positive_samples = sum(value > 0.0 for value in matrix)
    paired_advantage = fmean(m - s for m, s in zip(matrix, scalar))
    mean_retention = fmean(retention)
    no_go = (
        bind_count >= 30
        and mean_retention < 0.20
        and nonpositive_fragments >= 3
    )
    trigger = (
        positive_fragments >= 3
        and positive_samples >= 24
        and paired_advantage > 0.0
    )
    return {
        "schema": "cttn_shadow_analysis_v1",
        "decision": "NO-GO" if no_go else "TRIGGER" if trigger else "INCONCLUSIVE",
        "sample_count": len(samples),
        "bind_count": bind_count,
        "mean_retention": mean_retention,
        "mean_matrix_alignment": fmean(matrix),
        "mean_scalar_alignment": fmean(scalar),
        "paired_matrix_minus_scalar": paired_advantage,
        "positive_matrix_samples": positive_samples,
        "positive_fragment_means": positive_fragments,
        "nonpositive_fragment_means": nonpositive_fragments,
        "fragment_matrix_means": fragment_means,
        "no_go": no_go,
        "trigger": trigger,
    }


def analyze_records(records: Iterable[Mapping], *, expected_samples: int = 32) -> dict:
    return decide(shadow_samples(records), expected_samples=expected_samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tape", type=Path)
    parser.add_argument("--expected-samples", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.tape.read_text().splitlines() if line]
    result = analyze_records(records, expected_samples=args.expected_samples)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
