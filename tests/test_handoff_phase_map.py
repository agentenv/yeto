from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.handoff_phase_map import HandoffError, handoff_phase_map


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_source(tmp_path: Path, *, register_probe: bool = True) -> tuple[Path, Path]:
    source = tmp_path / "science" / "phase-map"
    probe = (
        source
        / "cells"
        / "cell-1"
        / "attempt-1"
        / "work"
        / "m4"
        / "syncer_probe"
        / "step-0001.bin"
    )
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"probe-capture" * 1024)
    metadata = source / "phase-map-manifest.json"
    metadata.write_text(json.dumps({"status": "sealed_results"}) + "\n")
    (source / "acquisition-seal.json").write_text("{}\n")
    (source / "phase-map.sha256").write_text(
        f"{_sha256(metadata)}  phase-map-manifest.json\n"
    )
    entries = [f"{_sha256(metadata)}  phase-map-manifest.json"]
    if register_probe:
        entries.append(f"{_sha256(probe)}  {probe.relative_to(source).as_posix()}")
    (source / "acquisition.sha256").write_text("\n".join(entries) + "\n")
    return source, probe


def test_handoff_hardlinks_closed_registered_probe_files(tmp_path):
    source, probe = _sealed_source(tmp_path)
    destination = tmp_path / "outer" / "phase-map"

    report = handoff_phase_map(source, destination)

    handed_probe = destination / probe.relative_to(source)
    assert handed_probe.read_bytes() == probe.read_bytes()
    assert handed_probe.stat().st_dev == probe.stat().st_dev
    assert handed_probe.stat().st_ino == probe.stat().st_ino
    assert (
        destination / "phase-map-manifest.json"
    ).stat().st_ino != (source / "phase-map-manifest.json").stat().st_ino
    assert report["probe_hardlink_files"] == 1
    assert report["probe_hardlink_bytes"] == probe.stat().st_size


def test_handoff_rejects_unregistered_probe_file(tmp_path):
    source, _probe = _sealed_source(tmp_path, register_probe=False)

    with pytest.raises(HandoffError, match="absent from acquisition"):
        handoff_phase_map(source, tmp_path / "outer" / "phase-map")
