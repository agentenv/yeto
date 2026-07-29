#!/usr/bin/env python3
"""Hash every file in the pinned SmolLM2-1.7B snapshot for V14 staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import day3_common as common


REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
DEFAULT_MODEL = Path(
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots"
) / REVISION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model
    if model.name != REVISION or not model.is_dir():
        raise SystemExit("V14 model snapshot/revision mismatch")
    files = {}
    for path in sorted(item for item in model.iterdir() if item.is_file()):
        resolved = path.resolve(strict=True)
        files[path.name] = {
            "bytes": resolved.stat().st_size,
            "sha256": common.sha256_file(resolved),
            "snapshot_entry_is_symlink": path.is_symlink(),
        }
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    if not required <= set(files):
        raise SystemExit(f"V14 model snapshot lacks required files: {sorted(required - set(files))}")
    payload = {
        "schema": "yeto_v14_model_proof_v1",
        "model": {
            "id": "HuggingFaceTB/SmolLM2-1.7B",
            "revision": REVISION,
            "path": str(model),
            "files": files,
            "canonical_inventory_sha256": common.canonical_sha256(files),
        },
    }
    common.write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": common.sha256_file(args.output),
                "files": len(files),
                "inventory": payload["model"]["canonical_inventory_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
