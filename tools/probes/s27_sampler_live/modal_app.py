"""S27 sampler-live: SGLang sampler weight updates during a generation, 1 x H100.

One container runs the ``sglang-server`` adapter on Qwen3-0.6B and replays
releases (two perturbed copies of the base weights) into rounds that are
generating:

* phase ``wait``: ``in_place_reload`` only. SGLang's update waits for the
  running requests (the E5 behavior: a release waits behind the round);
* phase ``live``: ``live_update`` (pause retract, update, continue).

Per update it records the ``load`` call time, the time left in the round when
the load returned, the pause segments, and per completion the versions and
token counts; after each round, a teacher-forced probe of a fixed text, so the
two phases can be compared for the same release directory.

    modal run --detach tools/probes/s27_sampler_live/modal_app.py --tag a

The entry spawns and returns (never kill a ``.remote()`` caller). Results go to
the Volume ``yeto-s27-sampler-live-logs`` under ``<tag>/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "5b9dc2a9e6e315d73dabfbc6faae428b59cc01aabec6189645a5c8c80d6e3721")
MODEL = "Qwen/Qwen3-0.6B"
GPU_NAME = "H100"

app = modal.App("yeto-s27-sampler-live")
logs = modal.Volume.from_name("yeto-s27-sampler-live-logs", create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file(str(HERE), f"{WORKDIR}/probe_modal_app.py", copy=False)
)

PROMPTS = 96
MAX_TOKENS = 1536
ROUNDS = 4
LOAD_AT_S = 3.0  # seconds after a round starts
PROBE = ("Question: Natalia sold clips to 48 of her friends in April, and then she sold half as "
         "many clips in May. How many clips did Natalia sell altogether?\nAnswer:",
         " In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.")


def _perturbed(base: Path, out: Path, seed: int) -> None:
    """A copy of the base weights with every float tensor scaled by (1 + 0.01 * N(0, 1))."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, out / f.name)
    gen = torch.Generator().manual_seed(seed)
    for f in sorted(base.glob("*.safetensors")):
        tensors = load_file(str(f))
        for name, t in tensors.items():
            if t.is_floating_point():
                noise = torch.randn(t.shape, generator=gen, dtype=torch.float32)
                tensors[name] = (t.float() * (1 + 0.01 * noise)).to(t.dtype)
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})


def _engine(base: Path, live: bool, log_path: Path):
    sys.path.insert(0, WORKDIR)
    from yeto.rl.adapters.sampler_engines.server import ServerConfig
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine

    log = open(log_path, "ab")  # noqa: SIM115 - the server's lifetime

    def spawn(argv, env):
        log.write((" ".join(argv) + "\n").encode())
        log.flush()
        return subprocess.Popen(list(argv), env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)

    config = ServerConfig(base_dir=base, card_model="H100", device_family="cuda", engine_version="image",
                          precision="bf16", quant_scheme="bf16", topology="tp1", max_model_len=4096,
                          python="python3", in_place_reload=True, start_timeout_s=900,
                          request_timeout_s=1800, extra_args=("--mem-fraction-static", "0.8"))
    return SGLangServerEngine(config, SGLangOptions(tp=1, live_update=live), spawn=spawn)


def _phase(name: str, base: Path, releases: list[Path], out: Path) -> dict:
    sys.path.insert(0, WORKDIR)
    from yeto.rl.engine.sampler_engine import SamplingParams

    engine = _engine(base, name == "live", out / f"sglang-{name}.log")
    started = time.monotonic()
    engine.start(None, 0)
    rows: dict = {"phase": name, "start_s": round(time.monotonic() - started, 2), "rounds": []}
    prompts = [f"Write a long, detailed story about a lighthouse keeper, number {i}." for i in range(PROMPTS)]
    sampling = SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, top_p=1.0)
    rows["probe_base"] = engine.score(*PROBE)
    for r in range(ROUNDS):
        version = r + 1
        release = releases[r % len(releases)]
        result: dict = {}
        began = time.monotonic()
        thread = threading.Thread(target=lambda: result.update(groups=engine.generate(prompts, sampling),
                                                               ended=time.monotonic()))
        thread.start()
        time.sleep(LOAD_AT_S)
        load_began = time.monotonic()
        report = engine.load(release, version)
        load_ended = time.monotonic()
        thread.join()
        completions = [c for g in result["groups"] for c in g]
        versions: dict = {}
        for c in completions:
            key = f"{c.weight_version}->{c.final_weight_version}"
            versions[key] = versions.get(key, 0) + 1
        rows["rounds"].append({
            "round": r, "release": release.name, "version": version,
            "round_s": round(result["ended"] - began, 3),
            "load_call_at_s": round(load_began - began, 3),
            "load_s": round(load_ended - load_began, 3),
            "load_returned_before_round_end_s": round(result["ended"] - load_ended, 3),
            "report": {"reload_mode": report.reload_mode, "pause_s": round(report.pause_s, 3),
                       **{k: v for k, v in report.detail.items() if k != "model_dir"}},
            "served_after": engine.served_version(),
            "completions": len(completions),
            "finish": {f: sum(1 for c in completions if c.finish_reason == f)
                       for f in {c.finish_reason for c in completions}},
            "response_tokens": sum(len(c.token_ids) for c in completions),
            "tokens_per_s": round(sum(len(c.token_ids) for c in completions) / (result["ended"] - began), 1),
            "versions": versions,
            "probe": engine.score(*PROBE),
        })
        print(json.dumps({k: v for k, v in rows["rounds"][-1].items() if k != "probe"}), flush=True)
    engine.stop()
    rows["last_exit"] = engine.last_exit
    return rows


@app.function(image=image, volumes={"/data/logs": logs}, gpu=f"{GPU_NAME}!:1", cpu=8, memory=65536,
              timeout=2400)
def run(tag: str) -> None:
    from huggingface_hub import snapshot_download

    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    (out / "gpu.txt").write_text(gpu + "\n")
    if GPU_NAME not in gpu:
        raise SystemExit(f"wrong GPU: {gpu}")
    import sglang

    meta = {"gpu": gpu, "sglang": sglang.__version__, "model": MODEL, "prompts": PROMPTS,
            "max_tokens": MAX_TOKENS, "rounds": ROUNDS, "load_at_s": LOAD_AT_S}
    base = Path(snapshot_download(MODEL))
    releases = [Path("/tmp/r1"), Path("/tmp/r2")]
    for seed, path in enumerate(releases, start=1):
        _perturbed(base, path, seed)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    logs.commit()
    result = {"meta": meta}
    try:
        for phase in ("wait", "live"):
            result[phase] = _phase(phase, base, releases, out)
            (out / "result.json").write_text(json.dumps(result, indent=1))
            logs.commit()
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=1))
        logs.commit()


@app.local_entrypoint()
def main(tag: str = "a") -> None:
    call = run.spawn(tag)
    print(f"spawned run {call.object_id}")
