"""S28 G-120B: make the text-120B base on the Volume be release r1-v0's files.

``DeltaMirror.materialize`` reads a delta release's base from the base release's
own directory in the store (``weights_dir(record)``), so the v0 weight files must
sit under ``policy-releases/<v0 path>/``. The base was materialized at
``xc120/base/nemotron-120b-text``; copying 225 GB again inside the Volume would
double the stored bytes, so this MOVES each file into the release directory
(``os.replace``, same volume, no bytes read).

It then checks every file the trainer's record lists against what is on the
Volume -- name present and size equal -- which is the check
``tools/probes/s27_xcloud/place_base_release.py`` does after copying. The
trainer's record is the authority: it was written on npu1 over the checkpoint
the trainer trains.

``LATEST.json`` is NOT written here. The interface's order is every file first,
then ``LATEST.json`` last, so the sampler never sees a release it cannot load;
the dev machine uploads it after this returns clean.

    modal run tools/probes/s28_g120/place_v0_release_app.py [--dry-run]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
MOUNT = "/root/yeto-checkpoint-store"
BASE_DIR = f"{MOUNT}/xc120/base/nemotron-120b-text"
RELEASE_DIR = f"{MOUNT}/xc120/policy-releases/r1-v0"

app = modal.App(os.environ.get("YETO_G120_APP", "yeto-s28-g120-placev0"))
xc = modal.Volume.from_name("yeto-s27-xc", create_if_missing=False)
image = modal.Image.from_registry(IMAGE).env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})


@app.function(image=image, volumes={MOUNT: xc}, cpu=4.0, memory=16384, timeout=3600)
def place(dry_run: bool = False) -> dict:
    release = Path(RELEASE_DIR)
    base = Path(BASE_DIR)
    record_path = release / "RELEASE.json"
    if not record_path.is_file():
        raise SystemExit(f"{record_path} is missing: upload the trainer's RELEASE.json first")
    record = json.loads(record_path.read_text())
    wanted: dict[str, int] = {k: int(v) for k, v in record["files"].items()}

    moved, already, missing = [], [], []
    for name in sorted(wanted):
        target = release / name
        if target.is_file():
            already.append(name)
            continue
        source = base / name
        if not source.is_file():
            missing.append(name)
            continue
        if not dry_run:
            os.replace(source, target)  # same volume: no bytes copied
        moved.append(name)

    if not dry_run:
        xc.commit()

    on_disk, wrong_size = {}, []
    for name, size in sorted(wanted.items()):
        target = release / name
        if not target.is_file():
            continue
        actual = target.stat().st_size
        on_disk[name] = actual
        if actual != size:
            wrong_size.append({"name": name, "record": size, "volume": actual})

    base_left = sorted(p.name for p in base.iterdir()) if base.is_dir() else []
    return {
        "release_path": record["path"],
        "policy_hash": record["policy_hash"],
        "version": record["version"],
        "record_files": len(wanted),
        "moved": len(moved),
        "already_present": len(already),
        "missing_from_both": missing,
        "present_on_volume": len(on_disk),
        "wrong_size": wrong_size,
        "total_bytes_on_volume": sum(on_disk.values()),
        "total_bytes_in_record": sum(wanted.values()),
        "base_dir_leftovers": base_left,
        "dry_run": bool(dry_run),
    }


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    result = place.remote(dry_run=dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    ok = (not result["missing_from_both"] and not result["wrong_size"]
          and result["present_on_volume"] == result["record_files"]
          and result["total_bytes_on_volume"] == result["total_bytes_in_record"])
    print("\nPASS: every file the record lists is on the Volume with the recorded size"
          if ok else "\nFAIL: see missing_from_both / wrong_size above")
    if not ok and not dry_run:
        raise SystemExit(1)
