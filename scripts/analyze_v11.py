#!/usr/bin/env python3
"""Run the frozen G11 ratio-transport analyzer after both truth sweeps drain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tonight85_analysis import analyze_v11, append_note, read_json, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-tonight85")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--note", type=Path, default=Path("/private/tmp/h200-tonight85-note.md")
    )
    args = parser.parse_args()
    readout = analyze_v11(
        read_json(args.manifest), read_json(args.predictions), args.result_root
    )
    write_json_atomic(args.output, readout)
    append_note(args.note, f"G11 VERDICT: {readout['gate']['verdict']}")
    print(
        json.dumps(
            {"verdict": readout["gate"]["verdict"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
