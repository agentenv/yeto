#!/usr/bin/env python3
"""Capture the bracketed, loss-blind fleet proof for a v9 stage handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


NODES = ("h200-n1", "h200-n2")
REMOTE = r"""
set -eu
echo V6_PROCESSES
pgrep -af '[r]un_slot_v6.py' 2>/dev/null \
  | awk 'index($0,"tmux new-session")==0' || true
echo V9_PROCESSES
pgrep -af '[r]un_slot_v9.py' 2>/dev/null || true
echo COMPUTE_PROCESSES
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo V6_EVIDENCE
find -L /root/yeto-results-v6 -path '*/attempt-*/evidence.json' -type f -print0 2>/dev/null \
  | xargs -0 -r jq -r 'select(.status=="COMPLETED")|.cell_id' | sort -u
echo V6_SLOTS
if [ -d /root/yeto-results-v6/_controller/slots-v6 ]; then
  for f in /root/yeto-results-v6/_controller/slots-v6/*.json; do
    [ -f "$f" ] || continue
    jq -r '[input_filename,.state,.completed,.failures,.queue_total] | @tsv' "$f"
  done
fi
echo V9_EVIDENCE
find -L /root/yeto-results-v9 -path '*/attempt-*/evidence.json' -type f -print0 2>/dev/null \
  | xargs -0 -r jq -r 'select(.status=="COMPLETED")|.cell_id' | sort -u
echo V9_STAGE_A_SLOTS
if [ -d /root/yeto-results-v9/_controller/slots-v9/stage_1p7b ]; then
  for f in /root/yeto-results-v9/_controller/slots-v9/stage_1p7b/*.json; do
    [ -f "$f" ] || continue
    jq -r '[input_filename,.state,.completed,.failures,.queue_total] | @tsv' "$f"
  done
fi
echo END
"""


def inspect_node(node: str) -> dict:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"root@{node}",
            REMOTE,
        ],
        capture_output=True,
        text=True,
    )
    sections: dict[str, list[str]] = {}
    current = None
    headers = {
        "V6_PROCESSES",
        "V9_PROCESSES",
        "COMPUTE_PROCESSES",
        "V6_EVIDENCE",
        "V6_SLOTS",
        "V9_EVIDENCE",
        "V9_STAGE_A_SLOTS",
        "END",
    }
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line in headers:
                current = line
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)

    def slots(name: str) -> list[dict]:
        records = []
        for line in sections.get(name, []):
            parts = line.split("\t")
            if len(parts) != 5:
                records.append({"raw": line, "state": "MALFORMED"})
            else:
                records.append(
                    {
                        "path": parts[0],
                        "state": parts[1],
                        "completed": None if parts[2] in ("", "null") else int(parts[2]),
                        "failures": None if parts[3] in ("", "null") else int(parts[3]),
                        "queue_total": None if parts[4] in ("", "null") else int(parts[4]),
                    }
                )
        return records

    return {
        "return_code": result.returncode,
        "stderr": result.stderr.strip(),
        "v6_processes": sections.get("V6_PROCESSES", []),
        "v9_processes": sections.get("V9_PROCESSES", []),
        "compute_processes": sections.get("COMPUTE_PROCESSES", []),
        "v6_completed_cell_ids": sections.get("V6_EVIDENCE", []),
        "v6_slots": slots("V6_SLOTS"),
        "v9_completed_cell_ids": sections.get("V9_EVIDENCE", []),
        "v9_stage_a_slots": slots("V9_STAGE_A_SLOTS"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage_1p7b", "stage_7b"), required=True)
    parser.add_argument("--smoke-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    records = {node: inspect_node(node) for node in NODES}
    errors = []
    for node, record in records.items():
        if record["return_code"] != 0:
            errors.append(f"{node} inspection failed: {record['stderr']}")
        if record["v9_processes"]:
            errors.append(f"{node} has {len(record['v9_processes'])} v9 controllers")
        if record["compute_processes"]:
            errors.append(
                f"{node} has {len(record['compute_processes'])} GPU compute processes"
            )

    if args.stage == "stage_1p7b":
        v6_cells = sorted(
            {
                cell
                for record in records.values()
                for cell in record["v6_completed_cell_ids"]
            }
        )
        if len(v6_cells) != 540:
            errors.append(f"v6 has {len(v6_cells)}/540 unique COMPLETED cells")
        for node, record in records.items():
            if record["v6_processes"]:
                errors.append(f"{node} still has v6 controllers")
            slots = record["v6_slots"]
            if len(slots) != 8 or any(slot["state"] != "DRAINED" for slot in slots):
                errors.append(f"{node} lacks eight exact v6 DRAINED slot states")
        prerequisite = {
            "kind": "v6_factorial_drain",
            "unique_completed_cells": len(v6_cells),
            "cell_ids": v6_cells,
            "required_slots_per_node": 8,
        }
    else:
        stage_a_ids = sorted(
            {
                cell
                for record in records.values()
                for cell in record["v9_completed_cell_ids"]
                if cell.startswith("v9-1p7b-")
            }
        )
        if len(stage_a_ids) != 16:
            errors.append(f"v9 1.7B has {len(stage_a_ids)}/16 unique COMPLETED cells")
        for node, record in records.items():
            slots = record["v9_stage_a_slots"]
            if len(slots) != 8 or any(slot["state"] != "DRAINED" for slot in slots):
                errors.append(f"{node} lacks eight exact v9 1.7B DRAINED slot states")
        smoke = None
        if args.smoke_proof is None or not args.smoke_proof.is_file():
            errors.append("7B stage requires the registered Qwen one-step smoke proof")
        else:
            smoke = read_json(args.smoke_proof)
            if (
                smoke.get("schema") != "yeto_outer_mup_v9_qwen_smoke_v1"
                or smoke.get("status") != "PASS"
                or smoke.get("model_revision")
                != manifest.get("models", {}).get("qwen2p5_7b", {}).get("revision")
            ):
                errors.append("Qwen smoke proof is not a bound PASS")
        prerequisite = {
            "kind": "v9_1p7b_drain_and_qwen_smoke",
            "unique_completed_cells": len(stage_a_ids),
            "cell_ids": stage_a_ids,
            "required_slots_per_node": 8,
            "smoke_proof_path": str(args.smoke_proof) if args.smoke_proof else None,
            "smoke_proof_sha256": (
                sha256_file(args.smoke_proof)
                if args.smoke_proof and args.smoke_proof.is_file()
                else None
            ),
            "smoke": smoke,
        }
    proof = {
        "schema": "yeto_outer_mup_v9_fleet_gate_v1",
        "stage": args.stage,
        "checked_at_utc": utc_now(),
        "checked_at_unix_s": time.time(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bracketed_process_patterns": ["[r]un_slot_v6.py", "[r]un_slot_v9.py"],
        "nodes": records,
        "prerequisite": prerequisite,
        "errors": errors,
        "status": "PASS" if not errors else "WAIT",
    }
    write_json_atomic(args.output.resolve(), proof)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": proof["status"],
                "errors": errors,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 2 if args.require_pass and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
