#!/usr/bin/env python3
"""Write deterministic SHA-256 inventories for the completed packet tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {
    ROOT / "SHA256SUMS",
    ROOT / "artifact-inventory.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path not in EXCLUDED
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    ]
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    inventory = {
        "schema": "yeto_best_paper_integration_packet_inventory_v1",
        "operator_label_commit": "5966e84",
        "source_commit": "8d58208cacafef12cb95f2642b4fa700531151b4",
        "file_count": len(rows),
        "files": rows,
    }
    (ROOT / "artifact-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    (ROOT / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    )


if __name__ == "__main__":
    main()
