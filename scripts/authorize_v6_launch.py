#!/usr/bin/env python3
"""Issue the immutable 30-hour v6 launch authority after fresh gate proof."""

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
WALL_SECONDS = 108_000


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    gate_path = args.gate_proof.resolve()
    manifest = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        errors.append("manifest schema mismatch")
    if manifest.get("status") != "REGISTERED" or len(manifest.get("cells", [])) != 900:
        errors.append("manifest is not the complete registered 900-cell grid")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(manifest_path):
        errors.append("manifest sidecar is missing or mismatched")
    if gate.get("schema") != "yeto_outer_mup_v6_gate_proof_v1" or gate.get("status") != "PASS":
        errors.append("gate proof is not PASS")
    if gate.get("v4", {}).get("unique_completed_cells") != 48:
        errors.append("gate proof lacks v4 48/48 completion")
    if any(gate.get("v4", {}).get("run_slot_processes", {}).get(node) for node in ("h200-n1", "h200-n2")):
        errors.append("gate proof records an active v4 controller")
    if not str(gate.get("v5", {}).get("verdict_line", "")).startswith("G5 VERDICT"):
        errors.append("gate proof lacks G5 VERDICT")
    now = time.time()
    checked = gate.get("checked_at_unix_s")
    if not isinstance(checked, (int, float)) or not 0 <= now - checked <= MAX_GATE_PROOF_AGE_SECONDS:
        errors.append("gate proof is absent, future-dated, or older than 10 minutes")
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
    remote_hash = remote.stdout.split()[0] if remote.returncode == 0 and remote.stdout.split() else None
    if remote_hash != head:
        errors.append(f"origin/{branch} is {remote_hash}, expected pushed HEAD {head}")
    if args.output.exists():
        errors.append("refusing to overwrite an existing launch authority")
    if errors:
        raise SystemExit("; ".join(errors))

    started = time.time()
    authority = {
        "schema": "yeto_outer_mup_v6_launch_authority_v1",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gate_proof_path": str(gate_path),
        "gate_proof_sha256": sha256_file(gate_path),
        "source_git_commit": head,
        "branch": branch,
        "contract": manifest["contract"],
        "wall_clock_start_unix_s": started,
        "hard_deadline_unix_s": started + WALL_SECONDS,
        "wall_ceiling_seconds": WALL_SECONDS,
    }
    write_json_atomic(args.output.resolve(), authority)
    print(json.dumps(authority, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
