"""S27 G-ES: write one full policy release for a rehearsal of the sampler's release path.

CPU container. It copies the Nano base checkpoint (Volume ``yeto-fn-models``) into a
``PolicyReleaseStore`` release under another root of the shared Volume
(default ``/xc-rehearsal/``, never the run's ``/xc/``), through the same Modal
Volume store the sampler reads (``yeto.cloud.modal_ckpt_store``): weights, then the
record, then ``LATEST.json``, one commit. The release is version ``--version``
with ``buffer_shards {"modal-0": 0}``; its weights are the base bytes, so the
sampler's in-place reload must serve the same answers as before.

    modal run tools/probes/s27_xcloud/rehearsal_release.py --root xc-rehearsal --version 1
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
BASE_DIR = "/models/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/bf77c317"
XC_VOLUME = "yeto-s27-xc"
STORE_MOUNT = "/root/yeto-checkpoint-store"
# files of the HF checkpoint the engine needs (weights, index, config, tokenizer, remote code)
KEEP_SUFFIXES = (".safetensors", ".json", ".py", ".jinja")

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-rehearsal-release"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6")
         .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                        ignore=["**/__pycache__", "**/*.pyc"]))


@app.function(image=image, volumes={"/models": models, STORE_MOUNT: xc}, cpu=8, memory=16384, timeout=3600)
def write_release(root: str, version: int) -> dict:
    sys.path.insert(0, WORKDIR)
    os.environ.update({"YETO_RL_STORE_IMPL": "yeto.cloud.modal_ckpt_store:store_from_env",
                       "YETO_RL_STORE_MODAL_VOLUME": XC_VOLUME, "YETO_RL_STORE_MODAL_MOUNT": STORE_MOUNT})
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import store_for

    if root.strip("/") in ("", "xc"):
        raise SystemExit("refusing to write into the run's root /xc/")
    base = Path(BASE_DIR)
    files = sorted(p for p in base.iterdir() if p.is_file() and p.name.endswith(KEEP_SUFFIXES)
                   and p.name != "MANIFEST.json")
    index_sha = hashlib.sha256((base / "model.safetensors.index.json").read_bytes()).hexdigest()

    def write(directory: Path) -> None:
        for p in files:
            shutil.copyfile(p, Path(directory) / p.name)

    store = PolicyReleaseStore(store_for(f"{STORE_MOUNT}/{root.strip('/')}"))
    began = time.monotonic()
    record = store.release(int(version), f"base-copy-{index_sha[:16]}", write, buffer_shards={"modal-0": 0},
                           origin={"by": "G-ES rehearsal", "base": BASE_DIR})
    out = {"root": root, "version": version, "path": None if record is None else record.path,
           "files": len(files), "bytes": sum(p.stat().st_size for p in files),
           "wall_s": round(time.monotonic() - began, 1), "timings": getattr(store, "last_timings", None)}
    print(json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(root: str = "xc-rehearsal", version: int = 1) -> None:
    print(json.dumps(write_release.remote(root, version)))
