#!/usr/bin/env python3
"""Run the prospectively frozen G13B Pythia/UltraChat regrid analyzer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tonight85_analysis import analyze_scan, append_note, read_json, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/root/yeto-results-v13b")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--note", type=Path, default=Path("/private/tmp/h200-g13fix-note.md")
    )
    args = parser.parse_args()

    # The original analysis core remains byte-for-byte frozen and only accepts
    # the original program label.  Relabel an in-memory copy so G13B uses the
    # exact same loader, quadratic acceptance rule, paired bootstrap, and gate.
    manifest = read_json(args.manifest)
    selected = [cell for cell in manifest.get("cells", []) if cell.get("program") == "v13b"]
    if len(selected) != 72:
        raise SystemExit(f"v13b manifest has {len(selected)} cells, expected 72")
    analysis_manifest = copy.deepcopy(manifest)
    for cell in analysis_manifest["cells"]:
        if cell.get("program") == "v13b":
            cell["program"] = "v13"
    readout = analyze_scan(analysis_manifest, "v13", args.result_root)
    readout["schema"] = "yeto_tonight85_v13b_readout_v1"
    readout["program"] = "v13b"
    readout["gate"]["name"] = "G13B"
    readout["analysis_identity"] = {
        "rule": "exact frozen v13 analysis core applied after in-memory program-label relabel only",
        "original_gate": "G13",
        "new_gate": "G13B",
    }
    write_json_atomic(args.output, readout)
    append_note(args.note, f"G13B VERDICT: {readout['gate']['verdict']}")
    print(
        json.dumps(
            {"verdict": readout["gate"]["verdict"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
