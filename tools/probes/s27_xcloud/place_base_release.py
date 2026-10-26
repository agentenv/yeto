"""S27 G-ES: put the base weights of the trainer's v0 release on the shared Volume.

The trainer's first release is a bf16 delta against its v0 release (``release_delta``).
The sampler rebuilds a delta from the base release directory named by
``delta.base_path`` (``DeltaMirror.materialize`` -> ``record_of``), so the Volume needs
that directory with every file. The mirror uploads only the v0 record
(``/xc/policy-releases/<path>/RELEASE.json``, a few KB); this CPU container copies the
files the record lists from the Nano base checkpoint (Volume ``yeto-fn-models``, the
same bytes as on npu1: ``s1-runs/s27-ascend/NANO-CKPT-SHA256.txt``) into that
directory, checks each size against the record and each sha256 against the
download manifest (and against ``file_sha256`` when the record has it), then commits.
It never writes ``LATEST.json``.

    modal run tools/probes/s27_xcloud/place_base_release.py --path r1-v0 [--root xc]
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import modal

BASE_DIR = "/models/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/bf77c317"
XC_VOLUME = "yeto-s27-xc"
MOUNT = "/root/yeto-checkpoint-store"

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-rel-base"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/models": models, MOUNT: xc},
              cpu=8, memory=8192, timeout=3600)
def place(path: str, root: str = "xc") -> dict:
    if "/" in path or not path.startswith("r") or "-v" not in path:
        raise SystemExit(f"bad release path {path!r}")
    xc.reload()
    target = Path(MOUNT) / root.strip("/") / "policy-releases" / path
    record = json.loads((target / "RELEASE.json").read_text())
    if record.get("delta") is not None or record.get("weights") != "full":
        raise SystemExit(f"{path} is not a full release record: weights={record.get('weights')}")
    files: dict[str, int] = record.get("files") or {}
    want_sha = record.get("file_sha256") or {}
    manifest = {f["path"]: f for f in json.loads((Path(BASE_DIR) / "MANIFEST.json").read_text())["files"]}
    base = Path(BASE_DIR)
    problems = []
    for name, size in files.items():
        src = base / name
        if not src.is_file():
            problems.append(f"{name}: not in the base checkpoint")
        elif src.stat().st_size != int(size):
            problems.append(f"{name}: base {src.stat().st_size} B, record {size} B")
    if problems:
        raise SystemExit("the record does not match the base checkpoint: " + "; ".join(problems[:10]))
    began = time.monotonic()

    def copy(name: str) -> dict:
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.is_file() and dst.stat().st_size == files[name]):
            shutil.copyfile(base / name, dst)
        digest = _sha256(dst)
        expect = want_sha.get(name) or (manifest.get(name) or {}).get("sha256")
        return {"name": name, "sha256": digest, "ok": expect is None or digest == expect, "checked": expect is not None}

    with cf.ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(copy, sorted(files)))
    copied_s = time.monotonic() - began
    bad = [r["name"] for r in rows if not r["ok"]]
    if bad:
        raise SystemExit(f"sha256 mismatch after the copy: {bad}")
    xc.commit()
    out = {"path": path, "files": len(rows), "bytes": sum(files.values()), "checked": sum(r["checked"] for r in rows),
           "copy_and_hash_s": round(copied_s, 1), "total_s": round(time.monotonic() - began, 1)}
    print(json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(path: str, root: str = "xc") -> None:
    print(json.dumps(place.remote(path, root)))
