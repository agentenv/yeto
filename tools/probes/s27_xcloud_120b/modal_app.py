"""S27 G-ES: Nemotron-3.5 Super VL 120B-A12B (bf16) rollout on Modal H200, smoke test.

One container (8 x H200) checks that the sampler island can serve the full
120B in bf16 and reload it in place, before it is wired to the Ascend trainer
(SESSION27/XCLOUD-120B-ROLLOUT.md). The engine is the sampler island's own
SGLang adapter (``yeto.rl.adapters.sampler_engines.sglang_server``), started and
reloaded exactly as the island does (``start`` / ``load`` with
``in_place_reload``); no Ray, no Miles.

Steps, each timed and written to ``result.json``:

1. the GPU names (all must be ``NVIDIA H200``, else exit before any load);
2. copy the checkpoint from the Volume ``yeto-fn-models`` to ``/dev/shm`` (the
   island loads every update from local memory) and SHA-256 every file of the
   copy against the download manifest (``MANIFEST.json`` on the Volume);
3. start SGLang TP8 on the copy; record the ready time and the GPU memory;
4. greedy answers on ``GREEDY_PROMPTS`` math prompts (correct / finite log-probs);
5. the sampling benchmark: ``BENCH_PROMPTS`` prompts x ``n`` samples at
   temperature 1 (tok/s, mean response length, truncation, reward);
6. ``update_weights_from_disk`` from the ``/dev/shm`` copy (version 1), then the
   greedy prompts again (the engine still serves the same answers);
7. optional: the same reload straight from the Volume (version 2).

    PLAN_ONLY=1 python tools/probes/s27_xcloud_120b/modal_app.py   # print the plan, no Modal call
    YETO_ES_APP=yeto-s27-es-1 modal run --detach tools/probes/s27_xcloud_120b/modal_app.py --tag s27-es-1/smoke

The entry spawns and returns (never kill a ``.remote()`` caller). Results go to
the Volume ``yeto-s27-es-logs`` under ``<tag>/``.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
# the image the training islands and spot samplers run (yeto.rl.adapters.miles.pins.MILES_NEXT_IMAGE)
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
MODEL_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
VOLUME_DIR = f"/models/hf/{MODEL.split('/')[-1]}/{MODEL_REVISION[:8]}"  # s1-runs/s27-120b/dl_hf_120b.py
SHM_DIR = "/dev/shm/ckpt-120b"
GPU_NAME = "NVIDIA H200"
GPUS = 8
PROMPTS_FILE = "dapo-head256.jsonl"  # first 256 rows of the S27 Nano DAPO-Math-17k task file (s27-dx-1)
PROMPTS_SRC = Path("/home/michael/work/s1-runs/s27-es") / PROMPTS_FILE
TIMEOUT_S = int(os.environ.get("YETO_ES_TIMEOUT_S", "4200"))
DEFAULTS = {"tp": 8, "greedy_prompts": 32, "bench_prompts": 128, "n": 8, "max_tokens": 2048,
            "max_model_len": 8192, "mem_fraction": 0.8, "reload_from_volume": True}


def plan(**overrides: object) -> dict:
    cfg = {**DEFAULTS, **overrides}
    if GPUS % int(cfg["tp"]):
        raise ValueError(f"TP {cfg['tp']} does not divide {GPUS} GPUs")
    return {"model": MODEL, "revision": MODEL_REVISION, "volume_dir": VOLUME_DIR, "shm_dir": SHM_DIR,
            "image": IMAGE, "gpu": f"H200!:{GPUS}", "timeout_s": TIMEOUT_S, "prompts": PROMPTS_FILE,
            "requests": int(cfg["bench_prompts"]) * int(cfg["n"]), **cfg}


if os.environ.get("PLAN_ONLY") == "1" and __name__ == "__main__":
    p = plan()
    if not PROMPTS_SRC.is_file():
        raise SystemExit(f"missing {PROMPTS_SRC}")
    print(json.dumps(p, indent=1))
    raise SystemExit(0)

import modal  # noqa: E402 - PLAN_ONLY runs without the Modal client

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-smoke"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
logs = modal.Volume.from_name("yeto-s27-es-logs", create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file(str(PROMPTS_SRC), f"{WORKDIR}/{PROMPTS_FILE}", copy=False)
)


class Recorder:
    """``result.json`` written (and the Volume committed) after every step."""

    def __init__(self, out: Path) -> None:
        self.out, self.data, self.t0 = out, {"steps": {}}, time.time()

    def step(self, name: str, **values: object) -> None:
        self.data["steps"][name] = {"t_s": round(time.time() - self.t0, 1), **values}
        print(f"[es] {name} {json.dumps(values)[:1500]}", flush=True)
        (self.out / "result.json").write_text(json.dumps(self.data, indent=1))
        try:
            logs.commit()
        except Exception as error:  # noqa: BLE001 - the next step commits again
            print(f"[es] volume commit failed: {error}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def copy_checkpoint(src: Path, dst: Path, workers: int = 16) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file())
    began = time.monotonic()
    with cf.ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda p: shutil.copyfile(p, dst / p.name), files))
    wall = time.monotonic() - began
    size = sum(p.stat().st_size for p in files)
    return {"files": len(files), "bytes": size, "wall_s": round(wall, 1), "gb_s": round(size / wall / 1e9, 3)}


def verify_checkpoint(copy: Path, manifest: dict, workers: int = 32) -> dict:
    want = {f["path"]: f["sha256"] for f in manifest["files"] if f.get("sha256")}
    began = time.monotonic()
    names = sorted(n for n in want if (copy / n).is_file())
    with cf.ThreadPoolExecutor(workers) as pool:
        got = dict(zip(names, pool.map(lambda n: _sha256(copy / n), names)))
    bad = [n for n in names if got[n] != want[n]]
    missing = sorted(set(want) - set(names))
    sample = {n: got[n] for n in ("config.json", "model.safetensors.index.json",
                                  "model-00001-of-00063.safetensors", "model-00032-of-00063.safetensors",
                                  "model-00063-of-00063.safetensors") if n in got}
    return {"checked": len(names), "bad": bad, "missing": missing, "wall_s": round(time.monotonic() - began, 1),
            "sha256": sample}


def render_prompts(model_dir: Path) -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    rows = []
    with open(f"{WORKDIR}/{PROMPTS_FILE}", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                row = json.loads(line)
                text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                rows.append({"prompt": text, "label": row["label"],
                             "task_id": (row.get("metadata") or {}).get("task_id")})
    return rows


def gpu_memory() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def score_round(rows: list[dict], completions: list[list], wall_s: float, gpus: int) -> dict:
    from yeto.rl.rewards.builtin import numeric_answer_reward
    from yeto.rl.rewards.types import Trajectory

    lens, rewards, finite, truncated, examples = [], [], True, 0, []
    for row, group in zip(rows, completions):
        for c in group:
            lens.append(len(c.token_ids))
            finite = finite and all(math.isfinite(x) for x in c.logprobs)
            truncated += c.finish_reason == "length"
            r = numeric_answer_reward(Trajectory(response=c.text, label=row["label"])).value
            rewards.append(float(r))
            if len(examples) < 3:
                examples.append({"task_id": row["task_id"], "label": row["label"], "reward": r,
                                 "finish": c.finish_reason, "tokens": len(c.token_ids),
                                 "tail": c.text[-300:], "logprob_mean": sum(c.logprobs) / max(1, len(c.logprobs))})
    tokens = sum(lens)
    return {"samples": len(lens), "output_tokens": tokens, "wall_s": round(wall_s, 1),
            "tok_s": round(tokens / wall_s, 1), "tok_s_per_gpu": round(tokens / wall_s / gpus, 1),
            "mean_len": round(tokens / max(1, len(lens)), 1), "max_len": max(lens, default=0),
            "truncated_frac": round(truncated / max(1, len(lens)), 4),
            "reward_mean": round(sum(rewards) / max(1, len(rewards)), 4), "logprobs_finite": finite,
            "examples": examples}


def greedy_tokens(completions: list[list]) -> list[tuple[int, ...]]:
    return [group[0].token_ids for group in completions]


def compare_greedy(a: list[list], b: list[list]) -> dict:
    same = sum(x[0].token_ids == y[0].token_ids for x, y in zip(a, b))
    prefix = []
    for x, y in zip(a, b):
        tx, ty = x[0].token_ids, y[0].token_ids
        k = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
        prefix.append(k)
    first = [abs(x[0].logprobs[0] - y[0].logprobs[0]) for x, y in zip(a, b) if x[0].logprobs and y[0].logprobs]
    return {"identical": same, "of": len(a), "mean_common_prefix": round(sum(prefix) / max(1, len(prefix)), 1),
            "max_first_token_logprob_diff": round(max(first, default=0.0), 6)}


@app.function(image=image, volumes={"/models": models, "/data/logs": logs}, gpu=f"H200!:{GPUS}", cpu=32,
              memory=655360, timeout=TIMEOUT_S)
def smoke(tag: str, cfg: dict) -> dict:
    sys.path.insert(0, WORKDIR)
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out)
    names = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip().splitlines()
    ok = len(names) == GPUS and all(line.split(",")[0].strip() == GPU_NAME for line in names)
    rec.step("gpu", names=names, ok=ok, plan=plan(**cfg))
    if not ok:
        raise SystemExit(f"wrong GPUs: {names}")
    nvml = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                             "--format=csv", "-l", "10"], stdout=open(out / "nvml.csv", "w"),
                            stderr=subprocess.STDOUT)
    src, shm = Path(VOLUME_DIR), Path(SHM_DIR)
    manifest = json.loads((src / "MANIFEST.json").read_text())
    rec.step("copy", **copy_checkpoint(src, shm), meminfo=Path("/proc/meminfo").read_text().splitlines()[:3])
    rec.step("verify", **verify_checkpoint(shm, manifest))

    import sglang

    from yeto.rl.adapters.sampler_engines.server import ServerConfig, _ignore_sigint
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine, sglang_topology
    from yeto.rl.engine.sampler_engine import SamplingParams

    server_log = open(out / "sglang.log", "ab")

    def spawn(argv, env):
        return subprocess.Popen(list(argv), env={**os.environ, **env}, stdout=server_log,
                                stderr=subprocess.STDOUT, preexec_fn=_ignore_sigint)

    tp = int(cfg["tp"])
    config = ServerConfig(base_dir=shm, card_model="H200", device_family="cuda", engine_version=sglang.__version__,
                          precision="bf16", quant_scheme="bf16", topology=sglang_topology(tp),
                          max_model_len=int(cfg["max_model_len"]), python=sys.executable, in_place_reload=True,
                          start_timeout_s=2400.0, request_timeout_s=3000.0,
                          extra_args=("--mem-fraction-static", str(cfg["mem_fraction"])))
    engine = SGLangServerEngine(config, SGLangOptions(tp=tp), spawn=spawn)
    rows = render_prompts(shm)
    (out / "prompt0.txt").write_text(rows[0]["prompt"])
    began = time.monotonic()
    report = engine.start(None, 0)
    rec.step("start", wall_s=round(time.monotonic() - began, 1), pause_s=round(report.pause_s, 1),
             gpu_mem_mib=gpu_memory(), sglang=sglang.__version__)
    try:
        greedy_rows = rows[:int(cfg["greedy_prompts"])]
        greedy = SamplingParams(max_tokens=int(cfg["max_tokens"]), temperature=0.0, n=1)
        began = time.monotonic()
        g0 = engine.generate([r["prompt"] for r in greedy_rows], greedy)
        rec.step("greedy0", **score_round(greedy_rows, g0, time.monotonic() - began, GPUS))

        bench_rows = rows[:int(cfg["bench_prompts"])]
        sampling = SamplingParams(max_tokens=int(cfg["max_tokens"]), temperature=1.0, top_p=1.0, n=int(cfg["n"]))
        began = time.monotonic()
        b0 = engine.generate([r["prompt"] for r in bench_rows], sampling)
        rec.step("bench", **score_round(bench_rows, b0, time.monotonic() - began, GPUS), gpu_mem_mib=gpu_memory())

        began = time.monotonic()
        load = engine.load(shm, 1)
        reload_s = time.monotonic() - began
        began = time.monotonic()
        g1 = engine.generate([r["prompt"] for r in greedy_rows], greedy)
        rec.step("reload_shm", reload_s=round(reload_s, 1), mode=load.reload_mode,
                 first_generate_s=round(time.monotonic() - began, 1), greedy=compare_greedy(g0, g1),
                 gpu_mem_mib=gpu_memory())
        if cfg.get("reload_from_volume"):
            began = time.monotonic()
            engine.load(src, 2)
            reload_s = time.monotonic() - began
            g2 = engine.generate([r["prompt"] for r in greedy_rows[:8]], greedy)
            rec.step("reload_volume", reload_s=round(reload_s, 1), greedy=compare_greedy(g0[:8], g2))
    finally:
        engine.stop()
        nvml.terminate()
        rec.step("done", wall_s=round(time.time() - rec.t0, 1))
    return rec.data


@app.local_entrypoint()
def main(tag: str, tp: int = 8, bench_prompts: int = 128, n: int = 8, max_tokens: int = 2048,
         reload_from_volume: bool = True) -> None:
    cfg = {**DEFAULTS, "tp": tp, "bench_prompts": bench_prompts, "n": n, "max_tokens": max_tokens,
           "reload_from_volume": reload_from_volume}
    plan(**cfg)
    call = smoke.spawn(tag, cfg)
    print(f"spawned smoke {call.object_id}")
