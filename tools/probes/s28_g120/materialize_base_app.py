"""S28 G-120: rebuild the text-only 120B base on a Modal Volume (G-120B).

The trainer on npu1 trains ``nemotron-120b-text``: the text subset of
NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16, materialized by
``yeto.rl.engine.text_subset_materialize`` (S27 G-EZ; 61 shards,
241,337,456,640 bytes). The Modal sampler island needs the same bytes as its
release version 0, because the delta releases rebuild against it.

npu1's uplink measured 2.3 MB/s (2026-10-27 00:5xZ), so copying 225 GB from
npu1 would take about 27 hours. Instead this job runs the same materialization
on Modal against the VL original that is already on the Volume
``yeto-fn-models``, and proves equality by comparing the manifest the
materializer writes (per-shard sha256 of all 61 shards, ``total_size``,
``source_index_sha256``, tensor counts, rule) against npu1's manifest.
The materializer copies tensor data byte for byte with the standard library,
so equal shard sha256 means equal bytes.

CPU only, no GPU.

    modal run tools/probes/s28_g120/materialize_base_app.py
    modal run tools/probes/s28_g120/materialize_base_app.py --verify-only
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
SRC = "/root/models/hf/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/fb0ee0bc"
DST = "/root/yeto-checkpoint-store/xc120/base/nemotron-120b-text"

app = modal.App(os.environ.get("YETO_G120_APP", "yeto-s28-g120-mat"))
xc = modal.Volume.from_name("yeto-s27-xc", create_if_missing=False)
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1", "PYTHONPATH": WORKDIR})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(
    image=image,
    volumes={"/root/yeto-checkpoint-store": xc, "/root/models": models},
    cpu=8.0,
    memory=32768,
    timeout=14400,
)
def materialize_base(verify_only: bool = False) -> dict:
    import sys

    sys.path.insert(0, WORKDIR)
    from yeto.rl.engine.text_subset_materialize import materialize, verify_tensor_bytes

    began = time.time()
    manifest_path = os.path.join(DST, "TEXT_SUBSET.json")
    if verify_only and os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print("verify-only: read existing manifest", flush=True)
    else:
        os.makedirs(DST, exist_ok=True)
        last = [0.0]

        def progress(message: str) -> None:
            now = time.time()
            if now - last[0] >= 20.0:
                last[0] = now
                print(f"[{now - began:8.1f}s] {message}", flush=True)

        manifest = materialize(SRC, DST, progress=progress)
        xc.commit()
        print(f"materialize done in {time.time() - began:.1f}s", flush=True)

    checked, details = verify_tensor_bytes(SRC, DST, sample=24)
    bad = [d for d in details if not d["equal"]]
    print(f"byte-compared {checked} tensors against the source, mismatches {len(bad)}",
          flush=True)

    on_disk = {}
    for name in sorted(os.listdir(DST)):
        path = os.path.join(DST, name)
        if os.path.isfile(path):
            on_disk[name] = os.path.getsize(path)
    return {
        "shards": manifest["shards"],
        "total_size": manifest["total_size"],
        "source_index_sha256": manifest["source_index_sha256"],
        "text_tensors": manifest["text_tensors"],
        "excluded_tensors": manifest["excluded_tensors"],
        "excluded_by_prefix": manifest["excluded_by_prefix"],
        "rule": manifest["rule"],
        "carried_files": manifest["carried_files"],
        "tensor_bytes_checked": checked,
        "tensor_bytes_mismatches": len(bad),
        "files_on_volume": on_disk,
        "seconds": round(time.time() - began, 1),
    }


@app.local_entrypoint()
def main(verify_only: bool = False, reference: str = "/tmp/g120b-ref/TEXT_SUBSET.json") -> None:
    result = materialize_base.remote(verify_only=verify_only)
    out = Path("/tmp/g120b-modal-manifest.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"manifest written to {out}")

    with open(reference) as fh:
        ref = json.load(fh)
    checks: list[tuple[str, bool, str]] = []
    for key in ("total_size", "source_index_sha256", "text_tensors", "excluded_tensors"):
        same = ref[key] == result[key]
        checks.append((key, same, f"npu1={ref[key]!r} modal={result[key]!r}"))
    checks.append(("excluded_by_prefix", ref["excluded_by_prefix"] == result["excluded_by_prefix"], ""))
    checks.append(("rule", ref["rule"] == result["rule"], ""))
    ref_shards, got_shards = ref["shards"], result["shards"]
    checks.append(("shard_count", len(ref_shards) == len(got_shards),
                   f"npu1={len(ref_shards)} modal={len(got_shards)}"))
    diff = [n for n in sorted(set(ref_shards) | set(got_shards))
            if ref_shards.get(n, {}).get("sha256") != got_shards.get(n, {}).get("sha256")]
    checks.append(("shard_sha256_all_equal", not diff, f"differing shards: {diff[:5]}"))
    checks.append(("tensor_bytes_mismatches_zero", result["tensor_bytes_mismatches"] == 0, ""))

    print("\n=== byte-identity checks against npu1 ===")
    for name, ok, note in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
    print(f"\nseconds: {result['seconds']}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)
    print("ALL-EQUAL: the Modal base is byte-identical to npu1's base")
