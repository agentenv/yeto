#!/usr/bin/env python3
"""Create the read-only v6 terminal/drain proof required before v7."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


PROCESS_PATTERN = "[r]un_slot_v6.py|[c]ompare_diloco.py|[y]eto.learner|[y]eto-syncer"
TERMINAL_NOTE_MARKERS = ("V6 DONE", "V6 COMPLETE", "G6 VERDICT:")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ssh(node: str, command: str) -> subprocess.CompletedProcess:
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        f"root@{node}",
        command,
    ]
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=stdout,
            stderr=(stderr + "\nSSH_CHECK_TIMEOUT").strip(),
        )


def process_check(node: str) -> dict:
    result = ssh(node, f"pgrep -af '{PROCESS_PATTERN}' || true")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {
        "command_kind": "pgrep_bracketed",
        "pattern": PROCESS_PATTERN,
        "return_code": result.returncode,
        "active_lines": lines,
        "stderr": result.stderr.strip(),
    }


def gpu_check(node: str) -> dict:
    result = ssh(
        node,
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits",
    )
    rows = []
    errors = []
    for line in result.stdout.splitlines():
        try:
            index, memory, utilization = [int(part.strip()) for part in line.split(",")]
            rows.append(
                {
                    "index": index,
                    "memory_used_mib": memory,
                    "utilization_percent": utilization,
                }
            )
        except Exception:
            errors.append(f"malformed GPU row: {line!r}")
    if result.returncode:
        errors.append(result.stderr.strip() or "nvidia-smi failed")
    if len(rows) != 8:
        errors.append(f"expected 8 GPU rows, found {len(rows)}")
    return {"return_code": result.returncode, "rows": rows, "errors": errors}


def slot_check(node: str) -> dict:
    script = r"""python3 - <<'PY'
import json
from pathlib import Path
root=Path('/root/yeto-results-v6/_controller/slots-v6')
rows=[]
if root.is_dir():
    for path in sorted(root.glob('*.json')):
        try:
            value=json.loads(path.read_text())
            rows.append({'name':path.name,'state':value.get('state'),'cell_id':value.get('cell_id'),'queue_index':value.get('queue_index'),'queue_total':value.get('queue_total')})
        except Exception as exc:
            rows.append({'name':path.name,'state':'INVALID','error':str(exc)})
print(json.dumps({'root_exists':root.is_dir(),'rows':rows},sort_keys=True))
PY"""
    result = ssh(node, script)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"root_exists": False, "rows": [], "parse_error": result.stdout}
    payload["return_code"] = result.returncode
    payload["stderr"] = result.stderr.strip()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorial-note",
        type=Path,
        default=Path("/private/tmp/h200-factorial-note.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    note_text = args.factorial_note.read_text()
    marker_lines = [
        line
        for line in note_text.splitlines()
        if any(marker in line for marker in TERMINAL_NOTE_MARKERS)
    ]
    nodes = {}
    errors = []
    checks = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for node in common.NODES:
            checks[node, "processes"] = executor.submit(process_check, node)
            checks[node, "gpus"] = executor.submit(gpu_check, node)
            checks[node, "slots"] = executor.submit(slot_check, node)
    for node in common.NODES:
        processes = checks[node, "processes"].result()
        gpus = checks[node, "gpus"].result()
        slots = checks[node, "slots"].result()
        if processes["return_code"] != 0 or processes["stderr"]:
            errors.append(f"{node}: bracketed process check failed")
        if processes["active_lines"]:
            errors.append(f"{node}: active v6/compute processes remain")
        if gpus["errors"]:
            errors.extend(f"{node}: {error}" for error in gpus["errors"])
        if any(
            row["memory_used_mib"] != 0 or row["utilization_percent"] != 0
            for row in gpus["rows"]
        ):
            errors.append(f"{node}: GPU occupancy is not zero")
        rows = slots.get("rows", [])
        if not slots.get("root_exists") or not rows:
            errors.append(f"{node}: v6 slot-state directory is absent or empty")
        if any(row.get("state") != "DRAINED" for row in rows):
            errors.append(f"{node}: not all v6 slot records are DRAINED")
        nodes[node] = {"processes": processes, "gpus": gpus, "slots": slots}
    if not marker_lines:
        errors.append("factorial note lacks a terminal v6 marker")
    proof = {
        "schema": "yeto_outer_mup_v7_v6_drain_proof_v1",
        "checked_at_utc": utc_now(),
        "factorial_note": {
            "path": str(args.factorial_note),
            "sha256": common.sha256_file(args.factorial_note),
            "terminal_marker_lines": marker_lines,
        },
        "process_pattern": PROCESS_PATTERN,
        "nodes": nodes,
        "errors": errors,
        "status": "PASS" if not errors else "WAIT",
    }
    common.write_json_atomic(args.output, proof)
    print(json.dumps({"status": proof["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 75


if __name__ == "__main__":
    raise SystemExit(main())
