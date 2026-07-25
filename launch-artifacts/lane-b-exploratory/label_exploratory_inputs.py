#!/usr/bin/env python3
"""Stamp an isolated seed bundle directory as EXPLORATORY without touching data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LABEL = "EXPLORATORY"


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stamp(path: Path) -> None:
    value = json.loads(path.read_text())
    stamped = {"label": LABEL}
    stamped.update({key: item for key, item in value.items() if key != "label"})
    if isinstance(stamped.get("seeds"), dict):
        stamped["seeds"] = {
            seed: {"label": LABEL, **{k: v for k, v in record.items() if k != "label"}}
            for seed, record in stamped["seeds"].items()
        }
    write_json_atomic(path, stamped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path("/root/yeto-data/outer-mup-explore"):
        raise SystemExit(f"refusing non-exploratory input root: {root}")
    manifests = sorted(root.glob("seed-*/input-manifest.json"))
    manifests.extend(sorted(root.glob("input-manifest*.json")))
    if len(list(root.glob("seed-*/train.jsonl"))) != 2:
        raise SystemExit("expected exactly two exploratory training files")
    for path in manifests:
        stamp(path)
    (root / "EXPLORATORY.md").write_text(
        "# EXPLORATORY — Lane B input bundles\n\n"
        "Seeds 401 and 409 only. The JSONL payloads remain byte-valid training data; "
        "this directory and all of its manifests are exploratory-only.\n"
    )
    print(json.dumps({"label": LABEL, "root": str(root), "manifests": len(manifests)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
