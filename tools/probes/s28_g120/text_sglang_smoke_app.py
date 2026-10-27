"""S28 G-120B: can SGLang serve the text-only 120B the Ascend trainer trains?

G-120-W2 §6 item 1 left this unverified: G-ES's 120B smoke loaded the VL
original (``NemotronH_Omni_Reasoning_V3``); the trainer trains the materialized
text subset (``nemotron_h``, 61 shards, 241,337,456,640 bytes). The whole Modal
agentic sampler island depends on this load working, so it is checked first and
on its own, before any Harbor or boat cost.

Cheaper than G-ES's smoke on purpose: it reads the checkpoint straight from the
Volume instead of copying 225 GB into ``/dev/shm`` first, and it does one greedy
round instead of a sampling benchmark. It uses the island's own engine adapter
(``sampler_engines.sglang_server``), started the way the island starts it.

Steps, each timed into ``result.json``:
1. GPU names (all must be H200) and the base directory's file count and bytes;
2. the tokenizer loads from the base directory and renders the chat template;
3. SGLang TP8 starts on the base directory; ready time and GPU memory;
4. greedy answers for a few prompts; finite log-probs; non-empty text.

    YETO_G120_APP=yeto-s28-g120-tsmoke modal run \
      tools/probes/s28_g120/text_sglang_smoke_app.py --tag s28-g120b/tsmoke
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
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
BASE_DIR = "/root/yeto-checkpoint-store/xc120/base/nemotron-120b-text"
GPU_NAME = "NVIDIA H200"
GPUS = 8
TIMEOUT_S = int(os.environ.get("YETO_G120_TIMEOUT_S", "5400"))
PROMPTS = [
    "What is 17 times 23? Answer with the number only.",
    "List the first five prime numbers, separated by commas.",
    "Write a one-line bash command that counts the lines of every .py file under the current directory.",
]

app = modal.App(os.environ.get("YETO_G120_APP", "yeto-s28-g120-tsmoke"))
xc = modal.Volume.from_name("yeto-s27-xc", create_if_missing=False)
logs = modal.Volume.from_name("yeto-s28-g120-logs", create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
)


def gpu_memory() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


@app.function(
    image=image,
    volumes={"/root/yeto-checkpoint-store": xc, "/data/logs": logs},
    gpu=f"H200:{GPUS}",
    cpu=32,
    memory=262144,
    timeout=TIMEOUT_S,
)
def smoke(tag: str, max_model_len: int = 65536, mem_fraction: float = 0.8,
          max_tokens: int = 128) -> dict:
    sys.path.insert(0, WORKDIR)
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    began_all = time.time()
    steps: list[dict] = []

    def step(name: str, **values: object) -> None:
        values["step"] = name
        values["at_s"] = round(time.time() - began_all, 1)
        steps.append(values)
        print(json.dumps(values, sort_keys=True), flush=True)
        (out / "result.json").write_text(json.dumps(steps, indent=2, sort_keys=True))

    names = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip().splitlines()
    gpus_ok = len(names) == GPUS and all(line.split(",")[0].strip() == GPU_NAME for line in names)
    base = Path(BASE_DIR)
    files = sorted(p for p in base.iterdir() if p.is_file())
    step("gpu", names=names, ok=gpus_ok, base_files=len(files),
         base_bytes=sum(p.stat().st_size for p in files),
         shards=len([p for p in files if p.name.endswith(".safetensors")]))
    if not gpus_ok:
        raise SystemExit(f"wrong GPUs: {names}")
    if not (base / "TEXT_SUBSET.json").is_file():
        raise SystemExit(f"{base} has no TEXT_SUBSET.json; materialize the base first")
    config_json = json.loads((base / "config.json").read_text())
    step("config", model_type=config_json.get("model_type"),
         architectures=config_json.get("architectures"))

    from transformers import AutoTokenizer

    began = time.monotonic()
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    rendered = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for p in PROMPTS
    ]
    step("tokenizer", wall_s=round(time.monotonic() - began, 1),
         vocab_size=len(tok), prompt0_tokens=len(tok(rendered[0])["input_ids"]))
    (out / "prompt0.txt").write_text(rendered[0])

    import sglang

    from yeto.rl.adapters.sampler_engines.server import ServerConfig, _ignore_sigint
    from yeto.rl.adapters.sampler_engines.sglang_server import (
        SGLangOptions, SGLangServerEngine, sglang_topology,
    )
    from yeto.rl.engine.sampler_engine import SamplingParams

    server_log = open(out / "sglang.log", "ab")

    def spawn(argv, env):
        return subprocess.Popen(list(argv), env={**os.environ, **env}, stdout=server_log,
                                stderr=subprocess.STDOUT, preexec_fn=_ignore_sigint)

    config = ServerConfig(base_dir=base, card_model="H200", device_family="cuda",
                          engine_version=sglang.__version__, precision="bf16", quant_scheme="bf16",
                          topology=sglang_topology(GPUS), max_model_len=int(max_model_len),
                          python=sys.executable, in_place_reload=True, start_timeout_s=3000.0,
                          request_timeout_s=3000.0,
                          extra_args=("--mem-fraction-static", str(mem_fraction)))
    engine = SGLangServerEngine(config, SGLangOptions(tp=GPUS), spawn=spawn)
    began = time.monotonic()
    try:
        report = engine.start(None, 0)
    except BaseException as exc:
        step("start_failed", error=f"{type(exc).__name__}: {exc}",
             wall_s=round(time.monotonic() - began, 1),
             sglang_log_tail=(out / "sglang.log").read_text(errors="replace")[-4000:])
        raise
    step("start", wall_s=round(time.monotonic() - began, 1), pause_s=round(report.pause_s, 1),
         gpu_mem_mib=gpu_memory(), sglang=sglang.__version__)
    try:
        params = SamplingParams(max_tokens=int(max_tokens), temperature=0.0, n=1)
        began = time.monotonic()
        completions = engine.generate(rendered, params)
        rows = []
        for prompt, group in zip(PROMPTS, completions):
            first = group[0]
            logprobs = list(getattr(first, "logprobs", None) or [])
            rows.append({
                "prompt": prompt,
                "text": (getattr(first, "text", "") or "")[:400],
                "tokens": len(getattr(first, "token_ids", None) or []),
                "logprobs": len(logprobs),
                "logprobs_finite": all(lp == lp and abs(lp) != float("inf") for lp in logprobs),
                "finish_reason": getattr(first, "finish_reason", None),
            })
        step("greedy", wall_s=round(time.monotonic() - began, 1), rows=rows,
             all_nonempty=all(r["text"].strip() for r in rows),
             all_logprobs_finite=all(r["logprobs_finite"] for r in rows),
             gpu_mem_mib=gpu_memory())
    finally:
        engine.stop()
        step("done", wall_s=round(time.time() - began_all, 1))
    logs.commit()
    return {"steps": steps}


@app.local_entrypoint()
def main(tag: str = "s28-g120b/tsmoke") -> None:
    call = smoke.spawn(tag)
    print(f"spawned text-120B SGLang smoke {call.object_id}")
    print(f"logs: modal app logs {app.app_id}")
