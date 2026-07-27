#!/usr/bin/env python3
"""Issue an immutable V8-mini authority bounded by wall and V9 priority gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
MAX_GATE_PROOF_AGE_SECONDS = 600
WALL_SECONDS = 21_600


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


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-proof", type=Path, required=True)
    parser.add_argument("--node-proof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    gate_path = args.gate_proof.resolve()
    manifest = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v8_launch_manifest_v1":
        errors.append("manifest schema mismatch")
    if manifest.get("stage") != "V8_PHASE_DIAGRAM":
        errors.append("manifest stage mismatch")
    if manifest.get("status") != "REGISTERED" or len(manifest.get("cells", [])) != 180:
        errors.append("manifest is not the complete registered 180-cell mini grid")
    if manifest.get("reuse", {}).get("reused_cell_count") != 0:
        errors.append("manifest differs from the zero-reuse registration")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(
        manifest_path
    ):
        errors.append("manifest sidecar is missing or mismatched")
    if gate.get("schema") != "yeto_outer_mup_v8_gate_proof_v2" or gate.get(
        "status"
    ) != "PASS":
        errors.append("v6-drain gate proof is not PASS")
    if not gate.get("v6", {}).get("all_slot_queues_drained"):
        errors.append("gate proof does not show all v6 queues drained")
    priority = gate.get("priority", {})
    if not priority.get("operator_fire_now"):
        errors.append("gate proof lacks the prospective FIRE NOW directive")
    if not priority.get("v9_not_active"):
        errors.append("gate proof does not show V9 priority work absent")
    priority_deadline = priority.get("deadline_unix_s")
    if not isinstance(priority_deadline, (int, float)):
        errors.append("gate proof lacks a numeric V9 priority deadline")
    elif priority_deadline <= time.time():
        errors.append("V9 priority deadline has already arrived")
    for node in ("h200-n1", "h200-n2"):
        record = gate.get("v6", {}).get("nodes", {}).get(node, {})
        if (
            record.get("active_v6_processes") != []
            or record.get("active_v9_processes") != []
            or not record.get("all_slots_drained")
        ):
            errors.append(f"gate proof does not show {node} idle and drained")
    now = time.time()
    checked = gate.get("checked_at_unix_s")
    if not isinstance(checked, (int, float)) or not 0 <= now - checked <= MAX_GATE_PROOF_AGE_SECONDS:
        errors.append("gate proof is absent, future-dated, or older than 10 minutes")

    node_proofs = {}
    for path in args.node_proof:
        proof_path = path.resolve()
        proof = json.loads(proof_path.read_text())
        node = proof.get("node")
        if proof.get("schema") != "yeto_outer_mup_v8_node_authority_v1" or proof.get(
            "status"
        ) != "PASS":
            errors.append(f"node proof is not PASS: {proof_path}")
        if proof.get("manifest_sha256") != sha256_file(manifest_path):
            errors.append(f"node proof binds another manifest: {proof_path}")
        if not isinstance(node, str) or node in node_proofs:
            errors.append(f"duplicate or missing node proof identity: {proof_path}")
        else:
            node_proofs[node] = {
                "path": str(proof_path),
                "sha256": sha256_file(proof_path),
            }
    if set(node_proofs) != {"h200-n1", "h200-n2"}:
        errors.append("exactly one PASS node proof per h200 node is required")

    head = git("rev-parse", "HEAD")
    if head != manifest.get("source", {}).get("git_commit"):
        errors.append("local HEAD differs from manifest source commit")
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        errors.append("tracked worktree is dirty")
    branch = manifest.get("source", {}).get("branch")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    remote_hash = (
        remote.stdout.split()[0]
        if remote.returncode == 0 and remote.stdout.split()
        else None
    )
    if remote_hash != head:
        errors.append(f"origin/{branch} is {remote_hash}, expected pushed HEAD {head}")
    if args.output.exists():
        errors.append("refusing to overwrite an existing launch authority")
    if errors:
        raise SystemExit("; ".join(errors))

    started = time.time()
    full_wall_deadline = started + WALL_SECONDS
    hard_deadline = min(full_wall_deadline, float(priority_deadline))
    authority = {
        "schema": "yeto_outer_mup_v8_launch_authority_v2",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gate_proof_path": str(gate_path),
        "gate_proof_sha256": sha256_file(gate_path),
        "node_proofs": node_proofs,
        "source_git_commit": head,
        "branch": branch,
        "contract": manifest["contract"],
        "wall_clock_start_unix_s": started,
        "full_wall_deadline_unix_s": full_wall_deadline,
        "priority_deadline_unix_s": float(priority_deadline),
        "hard_deadline_unix_s": hard_deadline,
        "registered_wall_ceiling_seconds": WALL_SECONDS,
        "effective_deadline_seconds": hard_deadline - started,
        "priority_stop_path": "/root/yeto-results-v8/_controller/V9_PRIORITY_YIELD",
    }
    write_json_atomic(args.output.resolve(), authority)
    print(json.dumps(authority, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
