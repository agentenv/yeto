#!/usr/bin/env python3
"""Run the frozen G12 heavy-ball monotonicity analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tonight85_analysis import analyze_scan, append_note, read_json, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-tonight85")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--note", type=Path, default=Path("/private/tmp/h200-tonight85-note.md")
    )
    args = parser.parse_args()
    readout = analyze_scan(read_json(args.manifest), "v12", args.result_root)
    write_json_atomic(args.output, readout)
    append_note(args.note, f"G12 VERDICT: {readout['gate']['verdict']}")
    print(
        json.dumps(
            {"verdict": readout["gate"]["verdict"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
