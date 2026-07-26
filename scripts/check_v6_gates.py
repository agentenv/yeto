#!/usr/bin/env python3
"""Create the loss-blind fleet gate proof required before any v6 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


NODES = ("h200-n1", "h200-n2")
REMOTE_COMMAND = r"""
set -eu
find -L /root/yeto-results-v4 -path '*/attempt-*/evidence.json' -type f -print0 2>/dev/null \
  | xargs -0 -r jq -r 'select(.status=="COMPLETED")|.cell_id' \
  | sort -u \
  | sed 's/^/CELL /'
pgrep -af '[r]un_slot_v4.py' 2>/dev/null | sed 's/^/PROC /' || true
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def inspect_node(node: str) -> dict:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", f"root@{node}", REMOTE_COMMAND],
        capture_output=True,
        text=True,
    )
    cells = []
    processes = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("CELL "):
                cells.append(line[5:])
            elif line.startswith("PROC "):
                processes.append(line[5:])
    return {
        "return_code": result.returncode,
        "stderr": result.stderr.strip(),
        "completed_cell_ids": sorted(set(cells)),
        "run_slot_processes": processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-note",
        type=Path,
        default=Path("/private/tmp/h200-snoofix-note.md"),
    )
    parser.add_argument(
        "--v4-note", type=Path, default=Path("/private/tmp/h200-v4-note.md")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    node_records = {node: inspect_node(node) for node in NODES}
    unique_cells = sorted(
        {
            cell
            for record in node_records.values()
            for cell in record["completed_cell_ids"]
        }
    )
    processes = {
        node: record["run_slot_processes"] for node, record in node_records.items()
    }
    verdict_line = None
    if args.v5_note.is_file():
        verdict_line = next(
            (
                line.strip()
                for line in args.v5_note.read_text().splitlines()
                if line.strip().startswith("G5 VERDICT")
            ),
            None,
        )
    errors = []
    for node, record in node_records.items():
        if record["return_code"] != 0:
            errors.append(f"{node} inspection failed: {record['stderr']}")
    if len(unique_cells) != 48:
        errors.append(f"v4 has {len(unique_cells)}/48 unique COMPLETED cells")
    for node in NODES:
        if processes[node]:
            errors.append(f"{node} still has {len(processes[node])} run_slot_v4 processes")
    if verdict_line is None:
        errors.append("v5 note lacks a G5 VERDICT line")

    checked_unix = time.time()
    proof = {
        "schema": "yeto_outer_mup_v6_gate_proof_v1",
        "checked_at_utc": utc_now(),
        "checked_at_unix_s": checked_unix,
        "status": "PASS" if not errors else "WAIT",
        "poll_cadence_seconds": 600,
        "v4": {
            "unique_completed_cells": len(unique_cells),
            "completed_cell_ids": unique_cells,
            "completed_by_node": {
                node: len(record["completed_cell_ids"])
                for node, record in node_records.items()
            },
            "run_slot_processes": processes,
            "node_inspection": node_records,
            "note_path": str(args.v4_note),
            "note_sha256": sha256_file(args.v4_note) if args.v4_note.is_file() else None,
        },
        "v5": {
            "note_path": str(args.v5_note),
            "note_sha256": sha256_file(args.v5_note) if args.v5_note.is_file() else None,
            "verdict_line": verdict_line,
        },
        "errors": errors,
    }
    write_json_atomic(args.output.resolve(), proof)
    print(
        json.dumps(
            {
                "status": proof["status"],
                "v4_completed": len(unique_cells),
                "v4_processes": {node: len(value) for node, value in processes.items()},
                "g5_verdict": verdict_line,
                "errors": errors,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    if args.require_pass and errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
