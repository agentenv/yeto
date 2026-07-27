#!/usr/bin/env python3
"""Prove V6 drain and an idle V9-priority window for a V8-mini launch."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


NODES = ("h200-n1", "h200-n2")
ACTIVE_PROCESS_PATTERN = (
    "[r]un_slot_v6.py|[c]ompare_diloco.py.*yeto-results-v6|[a]nalyze_v6.py|"
    "[r]un_slot_v9.py|[s]moke_v9_qwen.py|[f]reeze_v6_selection.py"
)
FIRE_NOW_MARKER = "## SUPERVISOR: FIRE NOW"
PRIORITY_DEADLINE_UTC = "2026-07-27T20:00:00+00:00"
PRIORITY_DEADLINE_UNIX_S = 1_785_182_400.0
MINIMUM_LAUNCH_WINDOW_SECONDS = 10_800
INPUT_MANIFEST_SHA256 = (
    "5f4235e56be5fc968227e02a6c9a6ebe57277d2736fb2947da14f7bd7f15a20b"
)
TOKEN_REPORT_SHA256 = (
    "de532769475ef116001748ffedc3824ab4292b93664cb0b2c5b27bdd48294d94"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ssh(node: str, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"root@{node}",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )


def classify_processes(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    active_v6 = []
    active_v9 = []
    ignored_stale_tmux = []
    for line in lines:
        fields = line.split(maxsplit=2)
        executable = fields[1] if len(fields) > 1 else ""
        # A tmux server retains the argv from the first session that created it.
        # It is not a live V6/V9 worker and may own unrelated sessions.
        if Path(executable).name == "tmux":
            ignored_stale_tmux.append(line)
            continue
        if (
            "run_slot_v6.py" in line
            or ("compare_diloco.py" in line and "yeto-results-v6" in line)
            or "analyze_v6.py" in line
        ):
            active_v6.append(line)
        if (
            "run_slot_v9.py" in line
            or "smoke_v9_qwen.py" in line
            or "freeze_v6_selection.py" in line
        ):
            active_v9.append(line)
    return active_v6, active_v9, ignored_stale_tmux


def inspect_node(node: str) -> dict:
    # The expression is deliberately bracketed so pgrep cannot match itself.
    script = f"""
set -u
printf 'PROCESSES_BEGIN\\n'
pgrep -af {shlex.quote(ACTIVE_PROCESS_PATTERN)} || true
printf 'PROCESSES_END\\n'
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

root=Path('/root/yeto-results-v6')
slots=root/'_controller'/'slots-v6'
records=[]
if slots.is_dir():
    for path in sorted(slots.glob('*.json')):
        try:
            value=json.loads(path.read_text())
            records.append({{'path':str(path),'state':value.get('state'),'cell_id':value.get('cell_id')}})
        except Exception as exc:
            records.append({{'path':str(path),'state':'INVALID','error':str(exc)}})
v8=Path('/root/yeto-results-v8')
scientific=[]
if v8.exists():
    scientific=[p.name for p in v8.iterdir() if p.name != '_controller']
input_path=Path('/root/yeto-data/outer-mup-v8/input-manifest.json')
token_path=Path('/root/yeto-data/outer-mup-v8/token-counts-smollm2.json')
stat=os.statvfs('/data')
print(json.dumps({{
 'v6_result_root_exists':root.exists(),
 'v6_slot_records':records,
 'v8_scientific_entries':scientific,
 'input_manifest_sha256':sha(input_path) if input_path.is_file() else None,
 'token_report_sha256':sha(token_path) if token_path.is_file() else None,
 'data_free_bytes':stat.f_bavail*stat.f_frsize,
}},sort_keys=True))
PY
"""
    try:
        result = ssh(node, script)
    except subprocess.TimeoutExpired:
        return {
            "return_code": None,
            "stderr": "SSH inspection timed out after 20 seconds",
            "active_v6_processes": None,
            "active_v9_processes": None,
            "all_slots_drained": False,
            "errors": ["remote inspection timed out"],
        }
    if result.returncode:
        return {
            "return_code": result.returncode,
            "stderr": result.stderr.strip(),
            "active_v6_processes": None,
            "active_v9_processes": None,
            "all_slots_drained": False,
            "errors": ["remote inspection failed"],
        }
    lines = result.stdout.splitlines()
    try:
        start = lines.index("PROCESSES_BEGIN")
        end = lines.index("PROCESSES_END")
        raw_processes = [line for line in lines[start + 1 : end] if line.strip()]
        payload = json.loads(lines[end + 1])
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        return {
            "return_code": result.returncode,
            "stderr": result.stderr.strip(),
            "active_v6_processes": None,
            "active_v9_processes": None,
            "all_slots_drained": False,
            "errors": [f"cannot parse remote inspection: {exc}"],
        }
    active_v6, active_v9, ignored_tmux = classify_processes(raw_processes)
    slots = payload["v6_slot_records"]
    all_slots_drained = (
        payload["v6_result_root_exists"]
        and len(slots) == 8
        and all(record.get("state") == "DRAINED" for record in slots)
    )
    errors = []
    if active_v6:
        errors.append("active v6 process")
    if active_v9:
        errors.append("active v9 priority process")
    if not all_slots_drained:
        errors.append("v6 has not drained all eight node-local slot queues")
    if payload["v8_scientific_entries"]:
        errors.append("v8 scientific result entries predate authority")
    if payload["input_manifest_sha256"] != INPUT_MANIFEST_SHA256:
        errors.append("v8 input manifest hash mismatch")
    if payload["token_report_sha256"] != TOKEN_REPORT_SHA256:
        errors.append("v8 token report hash mismatch")
    if payload["data_free_bytes"] < 1_000_000_000_000:
        errors.append("less than 1 TB free on /data")
    return {
        "return_code": result.returncode,
        "stderr": result.stderr.strip(),
        "bracketed_pgrep_pattern": ACTIVE_PROCESS_PATTERN,
        "raw_process_matches": raw_processes,
        "ignored_stale_tmux_matches": ignored_tmux,
        "active_v6_processes": active_v6,
        "active_v9_processes": active_v9,
        "v6_result_root_exists": payload["v6_result_root_exists"],
        "v6_slot_records": slots,
        "all_slots_drained": all_slots_drained,
        "v8_scientific_entries": payload["v8_scientific_entries"],
        "input_manifest_sha256": payload["input_manifest_sha256"],
        "token_report_sha256": payload["token_report_sha256"],
        "data_free_bytes": payload["data_free_bytes"],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator-note",
        type=Path,
        default=Path("/private/tmp/h200-phasediag-note.md"),
    )
    args = parser.parse_args()
    nodes = {node: inspect_node(node) for node in NODES}
    all_drained = all(record["all_slots_drained"] for record in nodes.values())
    no_v6 = all(record["active_v6_processes"] == [] for record in nodes.values())
    no_v9 = all(record["active_v9_processes"] == [] for record in nodes.values())
    no_errors = all(not record["errors"] for record in nodes.values())
    note_text = args.operator_note.read_text() if args.operator_note.is_file() else ""
    fire_now = any(
        line.strip() == FIRE_NOW_MARKER for line in note_text.splitlines()
    )
    checked = time.time()
    remaining = PRIORITY_DEADLINE_UNIX_S - checked
    enough_window = remaining >= MINIMUM_LAUNCH_WINDOW_SECONDS
    status = (
        "PASS"
        if all_drained and no_v6 and no_v9 and no_errors and fire_now and enough_window
        else "WAIT"
    )
    proof = {
        "schema": "yeto_outer_mup_v8_gate_proof_v2",
        "checked_at_utc": utc_now(),
        "checked_at_unix_s": checked,
        "loss_blind": True,
        "status": status,
        "v6": {
            "all_slot_queues_drained": all_drained,
            "nodes": nodes,
        },
        "priority": {
            "operator_note_path": str(args.operator_note.resolve()),
            "required_exact_marker": FIRE_NOW_MARKER,
            "operator_fire_now": fire_now,
            "v9_not_active": no_v9,
            "yield_on_v9_arrival": True,
            "deadline_utc": PRIORITY_DEADLINE_UTC,
            "deadline_unix_s": PRIORITY_DEADLINE_UNIX_S,
            "minimum_launch_window_seconds": MINIMUM_LAUNCH_WINDOW_SECONDS,
            "remaining_window_seconds": remaining,
            "enough_window": enough_window,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing proof: {args.output}")
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": status}, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
