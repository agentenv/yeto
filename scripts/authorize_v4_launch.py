#!/usr/bin/env python3
"""Create the immutable v4 launch authority and its 12-hour deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    if manifest.get("status") != "AUTHORIZED" or len(manifest.get("cells", [])) != 48:
        raise SystemExit("manifest is not the complete authorized v4 grid")
    wall = contract["wall_clock"]
    start = wall["start_unix_s"]
    deadline = wall["hard_deadline_unix_s"]
    if deadline - start != 43_200:
        raise SystemExit("contract does not encode exactly 12 hours")
    if time.time() >= deadline:
        raise SystemExit("registered v4 wall deadline already passed")
    authority = {
        "schema": "yeto_outer_mup_v4_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "wall_clock_start_event": wall["start_event"],
        "wall_clock_start_utc": wall["start_utc"],
        "wall_clock_start_unix_s": start,
        "hard_deadline_utc": wall["hard_deadline_utc"],
        "hard_deadline_unix_s": deadline,
        "ceiling_seconds": 43_200,
        "pilot_included_in_ceiling": True,
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps({"authority": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
