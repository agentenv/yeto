#!/usr/bin/env python3
"""Prove both nodes are v9-naive before the prospective prediction seal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import utc_now, write_json_atomic  # noqa: E402


NODES = ("h200-n1", "h200-n2")
REMOTE = r"""
set -eu
if [ -e /root/yeto-results-v9 ]; then echo RESULT_ROOT_EXISTS; fi
pgrep -af '[r]un_slot_v9.py' 2>/dev/null | sed 's/^/CONTROLLER /' || true
pgrep -af '[c]ompare_diloco.py|[y]eto (learner|syncer)' 2>/dev/null \
  | awk 'index($0,"/root/yeto-results-v9/"){print "RESULT_PROCESS " $0}' || true
find -L /root/yeto-results-v9 \( -name evidence.json -o -name results.jsonl -o -name attempt-start.json \) \
  -type f -print 2>/dev/null | sed 's/^/ARTIFACT /' || true
"""


def inspect(node: str) -> dict:
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
    root_exists = False
    controllers = []
    result_processes = []
    artifacts = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line == "RESULT_ROOT_EXISTS":
                root_exists = True
            elif line.startswith("CONTROLLER "):
                controllers.append(line.removeprefix("CONTROLLER "))
            elif line.startswith("RESULT_PROCESS "):
                result_processes.append(line.removeprefix("RESULT_PROCESS "))
            elif line.startswith("ARTIFACT "):
                artifacts.append(line.removeprefix("ARTIFACT "))
    return {
        "return_code": result.returncode,
        "stderr": result.stderr.strip(),
        "result_root_exists": root_exists,
        "controllers": controllers,
        "result_processes": result_processes,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing v9 preseal proof: {args.output}")
    nodes = {node: inspect(node) for node in NODES}
    errors = []
    for node, record in nodes.items():
        if record["return_code"] != 0:
            errors.append(f"{node} inspection failed: {record['stderr']}")
        if record["result_root_exists"]:
            errors.append(f"{node} already has /root/yeto-results-v9")
        if record["controllers"]:
            errors.append(f"{node} already has a v9 controller")
        if record["result_processes"]:
            errors.append(f"{node} already has a v9 result-bearing process")
        if record["artifacts"]:
            errors.append(f"{node} already has v9 attempt/result artifacts")
    proof = {
        "schema": "yeto_outer_mup_v9_preseal_proof_v1",
        "status": "PASS" if not errors else "FAIL",
        "checked_at_utc": utc_now(),
        "checked_at_unix_s": time.time(),
        "verification_loss_seen": False if not errors else None,
        "result_root_absent_on_both_nodes": not any(
            record["result_root_exists"] for record in nodes.values()
        ),
        "bracketed_process_patterns": [
            "[r]un_slot_v9.py",
            "[c]ompare_diloco.py|[y]eto (learner|syncer)",
        ],
        "nodes": nodes,
        "errors": errors,
    }
    write_json_atomic(args.output.resolve(), proof)
    print(
        json.dumps(
            {
                "status": proof["status"],
                "verification_loss_seen": proof["verification_loss_seen"],
                "errors": errors,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
