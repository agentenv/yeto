#!/usr/bin/env python3
"""Hand a sealed phase-map tree to an outer harness without duplicating probes.

The scientific runner may intentionally use a stable local path while the
outer harness uses an attempt-specific run directory. Probe captures can be
hundreds of gigabytes, so a recursive physical copy can exhaust the boot disk.
This helper copies ordinary sealed metadata and hardlinks only closed,
checksum-registered ``syncer_probe`` files on the same filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


class HandoffError(RuntimeError):
    pass


def _registered_paths(source: Path) -> set[str]:
    checksum = source / "acquisition.sha256"
    if not checksum.is_file() or checksum.is_symlink():
        raise HandoffError("sealed source lacks a regular acquisition.sha256")
    registered: set[str] = set()
    for line_number, line in enumerate(checksum.read_text().splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or not relative
        ):
            raise HandoffError(
                f"acquisition.sha256 line {line_number} is malformed"
            )
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise HandoffError("acquisition.sha256 contains an unsafe path")
        registered.add(path.as_posix())
    return registered


def _is_probe(relative: Path) -> bool:
    parts = relative.parts
    return any(
        part == "syncer_probe"
        and index >= 2
        and parts[index - 2 : index] == ("work", "m4")
        for index, part in enumerate(parts)
    )


def _require_sealed_source(source: Path) -> set[str]:
    for name in (
        "phase-map-manifest.json",
        "acquisition-seal.json",
        "acquisition.sha256",
        "phase-map.sha256",
    ):
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise HandoffError(f"sealed source lacks regular {name}")
    try:
        manifest = json.loads((source / "phase-map-manifest.json").read_text())
    except json.JSONDecodeError as exc:
        raise HandoffError("phase-map manifest is invalid JSON") from exc
    if manifest.get("status") not in (
        "sealed_acquisition_pending_teardown",
        "sealed_results",
    ):
        raise HandoffError("phase-map source is not sealed")
    if any(path.name.endswith(".tmp") for path in source.rglob("*")):
        raise HandoffError("sealed source still contains temporary files")
    return _registered_paths(source)


def handoff_phase_map(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination or source in destination.parents:
        raise HandoffError("destination must be outside the sealed source tree")
    if not source.is_dir() or source.is_symlink():
        raise HandoffError("source must be a regular directory")
    if destination.exists():
        raise HandoffError("destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise HandoffError("probe handoff requires source and destination on one filesystem")

    registered = _require_sealed_source(source)
    probe_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and _is_probe(path.relative_to(source))
    }
    missing = sorted(probe_paths - registered)
    if missing:
        raise HandoffError(
            "sealed probe files are absent from acquisition.sha256: "
            + ", ".join(missing[:3])
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.handoff-", dir=destination.parent)
    )
    linked_files = 0
    linked_bytes = 0
    copied_files = 0
    try:
        directories: list[tuple[Path, Path]] = []
        for root, dirnames, filenames in os.walk(source, followlinks=False):
            root_path = Path(root)
            relative_root = root_path.relative_to(source)
            target_root = temporary / relative_root
            target_root.mkdir(exist_ok=True)
            directories.append((root_path, target_root))
            for dirname in dirnames:
                candidate = root_path / dirname
                if candidate.is_symlink():
                    raise HandoffError(f"sealed source contains symlink: {candidate}")
            for filename in filenames:
                source_file = root_path / filename
                if source_file.is_symlink() or not source_file.is_file():
                    raise HandoffError(
                        f"sealed source contains a non-regular file: {source_file}"
                    )
                relative = source_file.relative_to(source)
                destination_file = temporary / relative
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                if _is_probe(relative):
                    os.link(source_file, destination_file)
                    source_stat = source_file.stat()
                    destination_stat = destination_file.stat()
                    if (
                        source_stat.st_dev != destination_stat.st_dev
                        or source_stat.st_ino != destination_stat.st_ino
                    ):
                        raise HandoffError(f"probe was not hardlinked: {relative}")
                    linked_files += 1
                    linked_bytes += source_stat.st_size
                else:
                    shutil.copy2(source_file, destination_file)
                    copied_files += 1
        for source_dir, destination_dir in reversed(directories):
            shutil.copystat(source_dir, destination_dir, follow_symlinks=False)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "schema": "yeto_phase_map_handoff_v1",
        "source": str(source),
        "destination": str(destination),
        "probe_hardlink_files": linked_files,
        "probe_hardlink_bytes": linked_bytes,
        "metadata_copied_files": copied_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        print(
            json.dumps(
                handoff_phase_map(args.source, args.destination),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, HandoffError) as exc:
        print(f"phase-map handoff error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
