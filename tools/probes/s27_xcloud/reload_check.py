"""S27 G-ES: is the in-place reload exact? (Nano, 2 x H200, SGLang TP2, the sampler's adapter).

The release-path rehearsal (s27-es-nr1) reloaded a byte copy of the base weights
(``/xc-rehearsal/policy-releases/r1-v1``) and the post-load probe moved by 0.0078
nats (mean of two pairs). This check separates the two explanations:

1. weights: SGLang's read-only ``POST /weights_checker {"action": "checksum"}``
   (per named parameter, per TP rank) before and after ``update_weights_from_disk``;
2. engine nondeterminism: the probe pairs scored ``REPEATS`` times on the base weights,
   with and without ``/flush_cache`` before each score, then the same after the reload.

    YETO_ES_APP=yeto-s27-es-nk1 modal run --detach tools/probes/s27_xcloud/reload_check.py --tag s27-es-nk1/check
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
# same values as sampler_app.py (not imported: the container gets only this file and yeto/)
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
BASE_DIR = "/models/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/bf77c317"
XC_VOLUME, STORE_MOUNT, GPU_NAME, TP = "yeto-s27-xc", "/root/yeto-checkpoint-store", "NVIDIA H200", 2
PROBE_PAIRS = [
    ["Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
     " In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72."],
    ["The capital of France is", " Paris. It is known for the Eiffel Tower and the Louvre museum."],
]

RELEASE = f"{STORE_MOUNT}/xc-rehearsal/policy-releases/r1-v1"
REPEATS = 3

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-reload-check"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)
logs = modal.Volume.from_name("yeto-s27-es-logs", create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"])
)


def checksums(engine) -> dict:
    began = time.monotonic()
    body = engine.call("POST", "/weights_checker", {"action": "checksum"})
    ranks = body.get("ranks") or [] if isinstance(body, dict) else []
    return {"success": body.get("success") if isinstance(body, dict) else None, "s": round(time.monotonic() - began, 1),
            "ranks": [{"per_gpu": r.get("per_gpu_checksum"), "tensors": r.get("checksums") or {}} for r in ranks],
            "raw_keys": sorted(body) if isinstance(body, dict) else str(body)[:300]}


def probes(engine, flush: bool) -> list[list[float]]:
    out = []
    for _ in range(REPEATS):
        row = []
        for prompt, continuation in PROBE_PAIRS:
            if flush:
                try:
                    engine.call("POST", "/flush_cache", None)
                except Exception as error:  # noqa: BLE001 - recorded; the score still runs
                    print(f"[nk] flush_cache failed: {error}", flush=True)
                    row.append(None)
            lp = engine.score(prompt, continuation)
            row.append(sum(lp) / len(lp))
        out.append(row)
    return out


def compare(a: dict, b: dict) -> dict:
    same = changed = missing = 0
    examples = []
    for ra, rb in zip(a["ranks"], b["ranks"]):
        for name, value in ra["tensors"].items():
            other = rb["tensors"].get(name)
            if other is None:
                missing += 1
            elif other == value:
                same += 1
            else:
                changed += 1
                if len(examples) < 5:
                    examples.append(name)
    return {"same": same, "changed": changed, "missing": missing, "examples": examples,
            "per_gpu_equal": [ra["per_gpu"] == rb["per_gpu"] for ra, rb in zip(a["ranks"], b["ranks"])]}


@app.function(image=image, volumes={"/models": models, STORE_MOUNT: xc, "/data/logs": logs}, gpu=f"H200:{TP}",
              cpu=16, memory=131072, timeout=1800)
def check(tag: str) -> dict:
    sys.path.insert(0, WORKDIR)
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, check=True).stdout.split("\n")
    names = [n.strip() for n in names if n.strip()]
    if len(names) != TP or any(n != GPU_NAME for n in names):
        raise SystemExit(f"wrong GPUs: {names}")
    import sglang

    from yeto.rl.adapters.sampler_engines.server import ServerConfig, _ignore_sigint
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine

    log = open(out / "sglang.log", "ab")

    def spawn(argv, env):
        return subprocess.Popen(list(argv), env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=_ignore_sigint)

    config = ServerConfig(base_dir=Path(BASE_DIR), card_model="H200", device_family="cuda",
                          engine_version=sglang.__version__, precision="bf16", quant_scheme="bf16",
                          topology=f"tp{TP}", max_model_len=4096, python=sys.executable, in_place_reload=True,
                          start_timeout_s=1800.0, request_timeout_s=1200.0,
                          extra_args=("--mem-fraction-static", "0.8"))
    engine = SGLangServerEngine(config, SGLangOptions(tp=TP), spawn=spawn)
    result: dict = {"gpus": names, "sglang": sglang.__version__, "release": RELEASE}

    def save() -> None:
        (out / "result.json").write_text(json.dumps(result, indent=1))
        logs.commit()

    began = time.monotonic()
    engine.start(None, 0)
    result["start_s"] = round(time.monotonic() - began, 1)
    try:
        c0 = checksums(engine)
        result["base"] = {"checksum_s": c0["s"], "success": c0["success"], "raw_keys": c0["raw_keys"],
                          "tensors": sum(len(r["tensors"]) for r in c0["ranks"]),
                          "probes_cached": probes(engine, flush=False), "probes_flushed": probes(engine, flush=True)}
        save()
        began = time.monotonic()
        engine.load(Path(RELEASE), 1)
        result["reload_s"] = round(time.monotonic() - began, 1)
        c1 = checksums(engine)
        result["reloaded"] = {"checksum_s": c1["s"], "success": c1["success"],
                              "tensors": sum(len(r["tensors"]) for r in c1["ranks"]),
                              "probes_cached": probes(engine, flush=False), "probes_flushed": probes(engine, flush=True)}
        result["checksum_compare"] = compare(c0, c1)
        save()
    finally:
        engine.stop()
        save()
    print("[nk] " + json.dumps({k: v for k, v in result.items() if k != "base"})[:3000], flush=True)
    return result


@app.local_entrypoint()
def main(tag: str) -> None:
    call = check.spawn(tag)
    print(f"spawned check {call.object_id}")
