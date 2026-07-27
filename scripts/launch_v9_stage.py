#!/usr/bin/env python3
"""Distribute an authorized v9 packet and start every registered stage queue."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


NODES = ("h200-n1", "h200-n2")
CACHE_ENV = {
    "HF_HOME": "/root/yeto-hf-cache",
    "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def remote_queue_argv(
    *,
    node: str,
    stage: str,
    slot_id: str,
    remote_root: str,
    attempt: int,
    retry_name: str | None,
) -> list[str]:
    argv = [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/run_slot_v9.py",
        "--manifest",
        f"{remote_root}/launch-manifest-v9.json",
        "--node-label",
        node,
        "--stage",
        stage,
        "--slot-id",
        slot_id,
        "--proof",
        f"{remote_root}/preflight-{stage}-{node}.json",
        "--launch-authority",
        f"{remote_root}/launch-authority-{stage}.json",
        "--attempt",
        str(attempt),
    ]
    if retry_name is not None:
        argv.extend(["--retry-authority", f"{remote_root}/{retry_name}"])
    return argv


def start_queue(
    *,
    node: str,
    stage: str,
    slot_id: str,
    remote_root: str,
    attempt: int,
    retry_name: str | None,
) -> dict:
    argv = remote_queue_argv(
        node=node,
        stage=stage,
        slot_id=slot_id,
        remote_root=remote_root,
        attempt=attempt,
        retry_name=retry_name,
    )
    log = f"{remote_root}/controller-{stage}-{slot_id}-attempt{attempt}.log"
    environment = ["env", *(f"{key}={value}" for key, value in CACHE_ENV.items())]
    shell = (
        "cd /root/yeto && nohup setsid "
        + shlex.join(environment)
        + " "
        + shlex.join(argv)
        + " > "
        + shlex.quote(log)
        + " 2>&1 < /dev/null & echo $!"
    )
    result = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"root@{node}",
            shell,
        ]
    )
    pid_text = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return {
        "node": node,
        "stage": stage,
        "slot_id": slot_id,
        "argv": argv,
        "remote_log": log,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "pid": int(pid_text) if pid_text.isdigit() else None,
    }


def copy_packet(
    *,
    node: str,
    remote_root: str,
    files: list[tuple[Path, str]],
) -> dict:
    mkdir = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            f"root@{node}",
            f"mkdir -p {shlex.quote(remote_root)}",
        ]
    )
    if mkdir.returncode:
        return {
            "node": node,
            "return_code": mkdir.returncode,
            "stderr": mkdir.stderr.strip(),
        }
    copied = []
    for source, name in files:
        result = run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                str(source),
                f"root@{node}:{remote_root}/{name}",
            ]
        )
        if result.returncode:
            return {
                "node": node,
                "return_code": result.returncode,
                "stderr": result.stderr.strip(),
                "copied": copied,
            }
        copied.append(
            {
                "name": name,
                "source": str(source.resolve()),
                "sha256": sha256_file(source),
            }
        )
    return {"node": node, "return_code": 0, "copied": copied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--node-proof", type=Path, action="append", required=True)
    parser.add_argument("--stage", choices=("stage_1p7b", "stage_7b"), required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument(
        "--remote-root",
        default="/root/yeto-results-v9/_controller/launch-v9",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing launcher record: {args.output}")
    manifest = read_json(args.manifest)
    authority = read_json(args.authority)
    if (
        manifest.get("schema") != "yeto_outer_mup_v9_launch_manifest_v1"
        or manifest.get("stage") != "V9_SEALED_SCALE"
        or len(manifest.get("cells", [])) != 28
    ):
        raise SystemExit("manifest is not the complete v9 launch")
    if (
        authority.get("schema") != "yeto_outer_mup_v9_launch_authority_v1"
        or authority.get("status") != "AUTHORIZED"
        or authority.get("stage") != args.stage
        or authority.get("manifest_sha256") != sha256_file(args.manifest)
    ):
        raise SystemExit("launch authority is invalid or binds another stage/manifest")
    if (args.attempt == 2) != (args.retry_authority is not None):
        raise SystemExit("attempt 2 requires, and attempt 1 forbids, retry authority")
    proof_by_node = {}
    for path in args.node_proof:
        proof = read_json(path)
        node = proof.get("node")
        if (
            node in proof_by_node
            or proof.get("status") != "PASS"
            or proof.get("stage") != args.stage
        ):
            raise SystemExit(f"invalid/duplicate node proof: {path}")
        proof_by_node[node] = path
    if set(proof_by_node) != set(NODES):
        raise SystemExit("node proofs must cover exactly h200-n1 and h200-n2")
    files_common = [
        (args.manifest, "launch-manifest-v9.json"),
        (
            args.manifest.with_suffix(args.manifest.suffix + ".sha256"),
            "launch-manifest-v9.json.sha256",
        ),
        (args.authority, f"launch-authority-{args.stage}.json"),
        (
            args.authority.with_suffix(args.authority.suffix + ".sha256"),
            f"launch-authority-{args.stage}.json.sha256",
        ),
    ]
    retry_name = None
    if args.retry_authority is not None:
        retry_name = f"retry-authority-{args.stage}.json"
        files_common.extend(
            [
                (args.retry_authority, retry_name),
                (
                    args.retry_authority.with_suffix(
                        args.retry_authority.suffix + ".sha256"
                    ),
                    retry_name + ".sha256",
                ),
            ]
        )
    copies = {}
    for node in NODES:
        files = files_common + [
            (proof_by_node[node], f"preflight-{args.stage}-{node}.json")
        ]
        copies[node] = copy_packet(node=node, remote_root=args.remote_root, files=files)
    if any(record["return_code"] for record in copies.values()):
        write_json_atomic(
            args.output.resolve(),
            {
                "schema": "yeto_outer_mup_v9_launcher_record_v1",
                "status": "COPY_FAILED_NO_CONTROLLERS_STARTED",
                "created_at_utc": utc_now(),
                "copies": copies,
            },
        )
        raise SystemExit("v9 packet distribution failed; no controller was started")
    slots = [
        (record["node"], record["slot_id"]) for record in authority["authorized_slots"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [
            executor.submit(
                start_queue,
                node=node,
                stage=args.stage,
                slot_id=slot_id,
                remote_root=args.remote_root,
                attempt=args.attempt,
                retry_name=retry_name,
            )
            for node, slot_id in slots
        ]
        launches = [future.result() for future in futures]
    success = all(record["return_code"] == 0 and record["pid"] for record in launches)
    result = {
        "schema": "yeto_outer_mup_v9_launcher_record_v1",
        "status": "STARTED" if success else "PARTIAL_START_REQUIRES_AUDIT",
        "created_at_utc": utc_now(),
        "stage": args.stage,
        "attempt": args.attempt,
        "manifest_sha256": sha256_file(args.manifest),
        "authority_sha256": sha256_file(args.authority),
        "copies": copies,
        "launches": launches,
    }
    write_json_atomic(args.output.resolve(), result)
    print(
        json.dumps(
            {"status": result["status"], "controllers": len(launches)}, sort_keys=True
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
