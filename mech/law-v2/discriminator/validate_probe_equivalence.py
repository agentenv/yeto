#!/usr/bin/env python3
"""Compare a discriminator adapter default-mode result with Lane-E output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(left, right, path: str, differences: list[dict]) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            differences.append({"path": path, "left": left, "right": right})
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_value = float(left)
        right_value = float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if left_value != right_value:
                differences.append(
                    {"path": path, "left": left_value, "right": right_value}
                )
            return
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(left_value), abs(right_value), 1.0)
        if absolute > 1e-6 and relative > 1e-6:
            differences.append(
                {
                    "path": path,
                    "left": left_value,
                    "right": right_value,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
            )
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append(
                {
                    "path": path,
                    "left_keys": sorted(left),
                    "right_keys": sorted(right),
                }
            )
            return
        for key in sorted(left):
            compare(left[key], right[key], f"{path}.{key}", differences)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append(
                {"path": path, "left_length": len(left), "right_length": len(right)}
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare(left_item, right_item, f"{path}[{index}]", differences)
        return
    if left != right:
        differences.append({"path": path, "left": left, "right": right})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-e", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lane_e = json.loads(args.lane_e.read_text(encoding="utf-8"))
    extension = json.loads(args.extension.read_text(encoding="utf-8"))
    lane_e.pop("runtime", None)
    extension.pop("runtime", None)
    differences: list[dict] = []
    compare(lane_e, extension, "$", differences)
    result = {
        "schema": "yeto_discriminator_adapter_equivalence_v1",
        "status": "PASS" if not differences else "FAIL",
        "tolerance": {"absolute": 1e-6, "relative": 1e-6},
        "lane_e_result": str(args.lane_e.resolve()),
        "extension_result": str(args.extension.resolve()),
        "differences": differences,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
