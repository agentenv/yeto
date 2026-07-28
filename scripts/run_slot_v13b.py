#!/usr/bin/env python3
"""Drain one hash-bound G13B queue in the isolated v13b checkout/result tree."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import tonight85_common as common
from run_slot_v3 import validate_cell


REPO = Path("/root/yeto-v13b")
RESULT_ROOT = Path("/root/yeto-results-v13b")


def kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def write(path: Path, value: object) -> None:
    common.write_json_atomic(path, value)


def verify_manifest(manifest_path: Path, manifest: dict, authority_path: Path) -> dict:
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(manifest_path):
        raise SystemExit("manifest SHA-256 sidecar missing or mismatched")
    if manifest.get("schema") != "yeto_v13b_launch_manifest_v1" or len(manifest.get("cells", [])) != 72:
        raise SystemExit("not the complete v13b manifest")
    authority = common.read_json(authority_path)
    authority_sidecar = authority_path.with_suffix(authority_path.suffix + ".sha256")
    if not authority_sidecar.is_file() or authority_sidecar.read_text().split()[0] != common.sha256_file(authority_path):
        raise SystemExit("launch authority SHA-256 sidecar missing or mismatched")
    if authority.get("schema") != "yeto_v13b_launch_authority_v1" or authority.get("status") != "AUTHORIZED":
        raise SystemExit("v13b launch authority is not authorized")
    if authority.get("manifest_sha256") != common.sha256_file(manifest_path):
        raise SystemExit("launch authority does not bind this manifest")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != manifest["source"]["git_commit"] or head != authority["source_git_commit"]:
        raise SystemExit("isolated checkout commit differs from manifest/authority")
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    if dirty:
        raise SystemExit("isolated v13b checkout is dirty")
    contract = REPO / manifest["registration"]["contract"]["path"]
    if common.sha256_file(contract) != manifest["registration"]["contract"]["sha256"]:
        raise SystemExit("v13b contract hash mismatch")
    return authority


def evidence_for_failure(cell: dict, status: str, failure: str, command_hash: str) -> dict:
    return {
        "schema": "yeto_outer_mup_cell_evidence_v1",
        "cell_id": cell["cell_id"],
        "validated_at_utc": common.utc_now(),
        "status": status,
        "failures": [failure],
        "expected": cell["expected"],
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": command_hash,
    }


def run_queue(manifest_path: Path, authority_path: Path, node: str, slot_id: str, attempt: int) -> int:
    manifest = common.read_json(manifest_path)
    verify_manifest(manifest_path, manifest, authority_path)
    queue = sorted(
        (
            cell for cell in manifest["cells"]
            if cell["stage"] == "v13b_regrid"
            and cell["slot_id"] == slot_id
            and cell["assignment"]["node"] == node
        ),
        key=lambda cell: cell["slot_queue_index"],
    )
    if not queue:
        raise SystemExit(f"no v13b queue for {node}/{slot_id}")
    status_path = RESULT_ROOT / "_controller" / "slots" / f"{slot_id}-a{attempt}.json"
    lock_path = RESULT_ROOT / "_controller" / "locks" / f"{slot_id}-a{attempt}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = failures = 0
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"slot already active: {slot_id}") from exc
        for queue_index, cell in enumerate(queue):
            attempt_root = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt}"
            evidence_path = attempt_root / "evidence.json"
            if evidence_path.is_file():
                evidence = common.read_json(evidence_path)
                if evidence.get("status") in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}:
                    completed += 1
                    continue
                raise SystemExit(f"refusing existing failed attempt: {evidence_path}")
            if attempt_root.exists():
                raise SystemExit(f"refusing to overwrite {attempt_root}")
            record = {"command": cell["command"], "command_hash": cell["command_hash"]} if attempt == 1 else next(
                item for item in cell["registered_retry_commands"] if item["attempt_number"] == attempt
            )
            command = record["command"]
            if common.canonical_sha256(command) != record["command_hash"]:
                raise SystemExit(f"{cell['cell_id']}: command hash changed")
            attempt_root.mkdir(parents=True)
            attempt_id = f"{cell['cell_id']}-a{attempt}-{uuid.uuid4()}"
            write(attempt_root / "attempt-start.json", {
                "schema": "yeto_v13b_attempt_start_v1", "attempt_id": attempt_id,
                "attempt_number": attempt, "cell_id": cell["cell_id"], "node": node,
                "gpus": cell["assignment"]["gpus"], "start_utc": common.utc_now(),
                "command": command, "command_hash": record["command_hash"],
                "manifest_sha256": common.sha256_file(manifest_path),
                "authority_sha256": common.sha256_file(authority_path),
                "git_commit": manifest["source"]["git_commit"],
            })
            write(status_path, {"schema": "yeto_v13b_slot_status_v1", "state": "RUNNING", "slot_id": slot_id, "node": node, "cell_id": cell["cell_id"], "queue_index": queue_index, "queue_total": len(queue), "completed": completed, "failures": failures, "updated_at_utc": common.utc_now()})
            env = dict(os.environ)
            env.update({"HF_DATASETS_CACHE": "/data/hf-datasets-cache", "HF_HUB_CACHE": "/root/yeto-hf-cache/hub", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
            log_path = attempt_root / "controller.stdout.log"
            started = time.monotonic()
            with log_path.open("wb") as log:
                process = subprocess.Popen(command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                timed_out = False
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= float(cell["timeout_minutes"]) * 60:
                        timed_out = True
                        kill_group(process)
                        break
                    write(status_path, {"schema": "yeto_v13b_slot_status_v1", "state": "RUNNING", "slot_id": slot_id, "node": node, "cell_id": cell["cell_id"], "elapsed_seconds": elapsed, "queue_index": queue_index, "queue_total": len(queue), "completed": completed, "failures": failures, "updated_at_utc": common.utc_now()})
                    time.sleep(30)
                return_code = process.wait()
            write(attempt_root / "attempt-end.json", {"schema": "yeto_v13b_attempt_end_v1", "attempt_id": attempt_id, "attempt_number": attempt, "cell_id": cell["cell_id"], "end_utc": common.utc_now(), "wall_seconds": time.monotonic() - started, "process_return_code": return_code, "timed_out": timed_out})
            if timed_out:
                evidence = evidence_for_failure(cell, "INFRA_FAILURE", "registered cell timeout", record["command_hash"])
            elif return_code:
                evidence = evidence_for_failure(cell, "INFRA_FAILURE", f"scientific process exited {return_code}", record["command_hash"])
            else:
                evidence = validate_cell({**cell, "source_git_commit": manifest["source"]["git_commit"]}, attempt_root, command)
            evidence.update({"attempt_id": attempt_id, "attempt_number": attempt, "git_commit": manifest["source"]["git_commit"], "retry_group_id": cell["retry_group_id"]})
            write(evidence_path, evidence)
            if evidence["status"] in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}:
                completed += 1
            else:
                failures += 1
        write(status_path, {"schema": "yeto_v13b_slot_status_v1", "state": "DRAINED", "slot_id": slot_id, "node": node, "attempt": attempt, "completed": completed, "failures": failures, "queue_total": len(queue), "updated_at_utc": common.utc_now()})
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--node-label", choices=("h200-n1", "h200-n2"), required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    return run_queue(args.manifest, args.authority, args.node_label, args.slot_id, args.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
