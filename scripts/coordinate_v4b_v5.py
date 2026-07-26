#!/usr/bin/env python3
"""Run collision-free node-local v4b-first queues with v5 GPU handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_controller(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write((f"START {utc_now()} {' '.join(command)}\n").encode())
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        handle.write((f"END {utc_now()} return_code={result.returncode}\n").encode())
        return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-label", choices=("h200-n1", "h200-n2"), required=True)
    parser.add_argument("--python", default="/root/yeto-venv/bin/python")
    parser.add_argument("--v4b-manifest", type=Path, required=True)
    parser.add_argument("--v4b-proof", type=Path, required=True)
    parser.add_argument("--v4b-authority", type=Path, required=True)
    parser.add_argument("--v5-manifest", type=Path, required=True)
    parser.add_argument("--v5-proof", type=Path, required=True)
    parser.add_argument("--v5-authority", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    v4b_base = [
        args.python,
        "/root/yeto/scripts/run_slot_g4b.py",
        "--manifest",
        str(args.v4b_manifest),
        "--node-label",
        args.node_label,
        "--proof",
        str(args.v4b_proof),
        "--launch-authority",
        str(args.v4b_authority),
    ]
    v5_base = [
        args.python,
        "/root/yeto/scripts/run_slot_v5.py",
        "--manifest",
        str(args.v5_manifest),
        "--node-label",
        args.node_label,
        "--proof",
        str(args.v5_proof),
        "--launch-authority",
        str(args.v5_authority),
    ]

    write_json_atomic(
        args.status,
        {
            "schema": "yeto_outer_mup_v4b_v5_coordinator_status_v1",
            "node": args.node_label,
            "state": "RUNNING",
            "started_at_utc": utc_now(),
            "partition": {"v4b_then_v5": list(range(6)), "v5_immediate": [6, 7]},
        },
    )

    def gpu_lane(gpu: int) -> dict:
        record = {"gpu": gpu, "v4b_return_code": None, "v5_return_code": None}
        if gpu < 6:
            record["v4b_return_code"] = run_controller(
                v4b_base + ["--gpu", str(gpu)],
                args.log_dir / f"gpu{gpu}-v4b.log",
            )
        record["v5_started_at_utc"] = utc_now()
        record["v5_return_code"] = run_controller(
            v5_base + ["--gpu", str(gpu)],
            args.log_dir / f"gpu{gpu}-v5.log",
        )
        record["finished_at_utc"] = utc_now()
        return record

    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(gpu_lane, gpu): gpu for gpu in range(8)}
        for future in as_completed(futures):
            records.append(future.result())
            write_json_atomic(
                args.status,
                {
                    "schema": "yeto_outer_mup_v4b_v5_coordinator_status_v1",
                    "node": args.node_label,
                    "state": "RUNNING",
                    "updated_at_utc": utc_now(),
                    "completed_gpu_lanes": sorted(records, key=lambda item: item["gpu"]),
                },
            )
    records.sort(key=lambda item: item["gpu"])
    failures = [
        record
        for record in records
        if record["v5_return_code"] != 0
        or (record["gpu"] < 6 and record["v4b_return_code"] != 0)
    ]
    write_json_atomic(
        args.status,
        {
            "schema": "yeto_outer_mup_v4b_v5_coordinator_status_v1",
            "node": args.node_label,
            "state": "DRAINED" if not failures else "DRAINED_WITH_FAILURES",
            "finished_at_utc": utc_now(),
            "gpu_lanes": records,
            "failures": failures,
        },
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
