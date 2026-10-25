"""S27 G-ES: the Modal sampler island of the cross-cloud RL check (Ascend trainer, Modal sampler).

Interface: ``s1-runs/s27-ascend/XCLOUD-INTERFACE.md`` (G-EQ; "G-ES 意见" for Nano).
The two sides share the Modal Volume ``yeto-s27-xc`` (root ``/xc/``); a mirror on
the dev machine moves the trainer's policy releases from npu1 to the Volume and
this sampler's groups from the Volume to npu1. The sampler is the unchanged
engine sampler island (``python -m yeto.rl.engine.engine_sampler``, SGLang
adapter, ``PolicyReleaseStore`` + ``FileSampleBuffer`` through
``yeto.cloud.modal_ckpt_store``); this file only builds its spec, mounts the
Volumes and stops the container when there is nothing to generate (the main
agent: containers run only while they generate).

* model: Nemotron-3 Nano 30B-A3B bf16, revision bf77c317, base weights (version 0)
  from the Volume ``yeto-fn-models`` (the same files as npu1:
  ``s1-runs/s27-ascend/NANO-CKPT-SHA256.txt``); 2 x H200, SGLang TP2 (the S27
  Nano engine shape), new releases loaded in place (``update_weights_from_disk``);
* buffer: shard ``modal-0`` (``sample-buffer/shards/modal-0.jsonl`` + payloads);
* prompts: ``/xc/prompts/train.jsonl`` (S27 Nano DAPO-Math-17k rows), rendered with the
  checkpoint's chat template, ``enable_thinking=False``; 16 prompts x 8 samples per
  round, 2048 response tokens, temperature 1; reward ``numeric_answer_reward``.

Stop rules (the wrapper writes the sampler's reclaim marker; the sampler leaves after
the round in progress, as PREEMPTED, and writes no data end):

1. ``backlog``: the sampler reported ``rl_sampler_backlog_full`` and started no round
   for ``idle_s`` seconds (nothing to do until the trainer reads or releases);
2. ``version_groups``: ``stop_groups`` groups were written at policy version
   ``>= stop_version`` (phase B: enough groups of the new weights);
3. ``wall``: ``max_s`` seconds after the container started.

``--store`` (default ``/root/yeto-checkpoint-store/xc``): another root on the same Volume
for a rehearsal of the release path (``rehearsal_release.py`` writes a release there).

    PLAN_ONLY=1 python tools/probes/s27_xcloud/sampler_app.py
    YETO_ES_APP=yeto-s27-es-n1 modal run --detach tools/probes/s27_xcloud/sampler_app.py --tag s27-es-n1/a

The entry spawns and returns (never kill a ``.remote()`` caller). Logs go to the
Volume ``yeto-s27-es-logs`` under ``<tag>/``: ``spec.json``, ``gpu.txt``,
``sampler.log``, ``events.<incarnation>.jsonl`` (the sampler tape), ``nvml.csv``,
``result.json``. Every tape line is also printed as ``[xc-tape] <json>``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
# the image of the S27 Nano islands and spot samplers (yeto.rl.adapters.miles.pins.MILES_NEXT_IMAGE)
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
BASE_DIR = f"/models/hf/{MODEL.split('/')[-1]}/{MODEL_REVISION[:8]}"  # s1-runs/s27-gr/dl_hf_nano.py
XC_VOLUME = "yeto-s27-xc"
STORE_MOUNT = "/root/yeto-checkpoint-store"  # yeto.launch.storage.MODAL_CHECKPOINT_STORE_MOUNT
STORE = f"{STORE_MOUNT}/xc"
PROMPTS = f"{STORE}/prompts/train.jsonl"
SHARD = "modal-0"
GPU_NAME = "NVIDIA H200"
TP = 2
RECLAIM_MARKER = "/tmp/yeto-reclaim-requested"  # sampler_resume.DEFAULT_RECLAIM_MARKER
TIMEOUT_S = int(os.environ.get("YETO_ES_TIMEOUT_S", "3600"))
DEFAULTS = {"store": STORE, "groups_per_round": 16, "samples_per_group": 8, "max_tokens": 2048, "max_backlog_groups": 64,
            "num_rollouts": 600, "idle_s": 60.0, "stop_version": -1, "stop_groups": 0, "max_s": 3000.0,
            "poll_s": 20.0, "mem_fraction": 0.8}
# Post-load probe (as the spot sampler): teacher-forced continuations; a wrong release moves
# their mean log-prob by several nats, a few RL steps at lr 1e-6 by far less than 0.5.
PROBE_PAIRS = [
    ["Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
     " In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72."],
    ["The capital of France is", " Paris. It is known for the Eiffel Tower and the Louvre museum."],
]


def spec(prompts: str, out: str, *, engine_version: str, cfg: dict) -> dict:
    server = {"base_dir": BASE_DIR, "card_model": "H200", "device_family": "cuda",
              "engine_version": engine_version, "precision": "bf16", "quant_scheme": "bf16",
              "topology": f"tp{TP}", "max_model_len": 4096, "python": sys.executable, "in_place_reload": True,
              "start_timeout_s": 1800.0, "request_timeout_s": 2400.0,
              "extra_args": ["--mem-fraction-static", str(cfg["mem_fraction"])]}
    return {
        "sampler_id": SHARD,
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                   "server": server, "options": {"tp": TP}},
        "sampling": {"max_tokens": int(cfg["max_tokens"]), "temperature": 1.0, "top_p": 1.0,
                     "n": int(cfg["samples_per_group"])},
        "num_rollouts": int(cfg["num_rollouts"]), "max_backlog_groups": int(cfg["max_backlog_groups"]),
        "max_prompt_tokens": 1024, "base_version": 0, "base_dir": BASE_DIR, "poll_s": float(cfg["poll_s"]),
        "store": str(cfg["store"]), "buffer_shards": [SHARD],
        "prompts": {"path": prompts, "input_key": "prompt", "label_key": "label",
                    "groups_per_round": int(cfg["groups_per_round"]), "format": "text"},
        "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles",
        # bf16 release: no expert count; raw GSM8K text is not the chat format, so a missing
        # answer is recorded only; empty / unreadable output and the probe jump still fail
        "checks": {"expected_expert_matrices": 0, "require_fixed_answers": False, "fixed_max_tokens": 512,
                   "probe_pairs": PROBE_PAIRS, "max_probe_jump": 0.5},
        "event_tape": f"{out}/events.jsonl",
    }


def plan(**overrides: object) -> dict:
    cfg = {**DEFAULTS, **overrides}
    return {"model": MODEL, "revision": MODEL_REVISION, "base_dir": BASE_DIR, "volume": XC_VOLUME,
            "store": cfg["store"], "shard": SHARD, "image": IMAGE, "gpu": f"H200:{TP}", "timeout_s": TIMEOUT_S,
            "spec": spec("/tmp/xc-prompts.jsonl", "/data/logs/<tag>", engine_version="<sglang>", cfg=cfg),
            "stop": {k: cfg[k] for k in ("idle_s", "stop_version", "stop_groups", "max_s")}}


if os.environ.get("PLAN_ONLY") == "1" and __name__ == "__main__":
    print(json.dumps(plan(), indent=1))
    raise SystemExit(0)

import modal  # noqa: E402 - PLAN_ONLY runs without the Modal client

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-sampler"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)
logs = modal.Volume.from_name("yeto-s27-es-logs", create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
)


def render_prompts(src: str, dst: str) -> int:
    """The prompt rows as text: the base checkpoint's chat template with thinking off
    (the S27 Nano runs' ``--apply-chat-template-kwargs {"enable_thinking": false}``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    rows = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["prompt"] if isinstance(row["prompt"], list) else [{"role": "user", "content": row["prompt"]}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            fout.write(json.dumps({"prompt": text, "label": row["label"],
                                   "task_id": (row.get("metadata") or {}).get("task_id")}) + "\n")
            rows += 1
    return rows


class StopRules:
    """Reads the tape lines and decides when to write the reclaim marker (module doc)."""

    def __init__(self, cfg: dict, began: float) -> None:
        self.cfg, self.began = cfg, began
        self.full_since: float | None = None
        self.version_groups = 0
        self.reason: str | None = None

    def see(self, record: dict) -> None:
        event = record.get("event")
        if event == "rl_sampler_backlog_full":
            self.full_since = time.time()
        elif event in ("rl_sampler_admitted", "rl_sampler_round"):
            self.full_since = None
        if event == "rl_sampler_round" and int(record.get("policy_version", -1)) >= int(self.cfg["stop_version"]):
            self.version_groups += int(record.get("groups") or 0)

    def check(self) -> str | None:
        now = time.time()
        if self.full_since is not None and now - self.full_since >= float(self.cfg["idle_s"]):
            return "backlog"
        if int(self.cfg["stop_groups"]) > 0 and self.version_groups >= int(self.cfg["stop_groups"]):
            return "version_groups"
        if now - self.began >= float(self.cfg["max_s"]):
            return "wall"
        return None


def _watch(out: Path, rules: StopRules, stop: threading.Event) -> None:
    """Print new tape lines, apply the stop rules, commit the logs Volume every 30 s."""
    offsets: dict[Path, int] = {}
    last_commit = 0.0
    while True:
        for tape in sorted(out.glob("events.*.jsonl")):
            try:
                with tape.open("rb") as fh:
                    fh.seek(offsets.get(tape, 0))
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n")
            if end < 0:
                continue
            offsets[tape] = offsets.get(tape, 0) + end + 1
            for line in data[:end].split(b"\n"):
                text = line.decode("utf-8", "replace")
                print("[xc-tape] " + text[:1500], flush=True)
                try:
                    rules.see(json.loads(text))
                except ValueError:
                    pass
        if rules.reason is None:
            reason = rules.check()
            if reason is not None:
                rules.reason = reason
                Path(RECLAIM_MARKER).write_text(json.dumps({"by": "wrapper", "reason": reason,
                                                            "time": time.time()}) + "\n")
                print(f"[xc] stop rule {reason}: reclaim marker written", flush=True)
        if time.monotonic() - last_commit >= 30.0:
            try:
                logs.commit()
            except Exception as error:  # noqa: BLE001 - retried on the next pass
                print(f"[xc] logs commit failed: {error}", flush=True)
            last_commit = time.monotonic()
        if stop.wait(5.0):
            return


def _mark_on_signal() -> None:
    """SIGINT/SIGTERM of this function (``modal app stop``) write the reclaim marker."""
    import signal

    def mark(signum: int, _frame: object) -> None:
        try:
            Path(RECLAIM_MARKER).write_text(json.dumps({"signal": int(signum), "time": time.time()}) + "\n")
        except OSError:
            pass
        print(f"[xc] signal {signum}: reclaim marker written", flush=True)

    for name in ("SIGINT", "SIGTERM"):
        try:
            signal.signal(getattr(signal, name), mark)
        except ValueError:
            print(f"[xc] {name} handler not installed (not the main thread)", flush=True)


# Modal has no "H200!" form (H200 has no upgrade path); the function checks the GPU names itself.
@app.function(image=image, volumes={"/models": models, STORE_MOUNT: xc, "/data/logs": logs}, gpu=f"H200:{TP}",
              cpu=16, memory=131072, timeout=TIMEOUT_S)
def sampler(tag: str, cfg: dict) -> int:
    began = time.time()
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip()
    (out / "gpu.txt").write_text(names + "\n")
    lines = names.splitlines()
    if len(lines) != TP or any(line.split(",")[0].strip() != GPU_NAME for line in lines):
        logs.commit()
        raise SystemExit(f"wrong GPUs: {names}")
    import sglang

    prompts = "/tmp/xc-prompts.jsonl"
    rows = render_prompts(PROMPTS, prompts)
    sp = spec(prompts, str(out), engine_version=sglang.__version__, cfg=cfg)
    (out / "spec.json").write_text(json.dumps(sp, indent=1))
    meta = {"tag": tag, "gpu": names, "sglang": sglang.__version__, "prompt_rows": rows, "cfg": cfg,
            "container_started_unix": began, "setup_s": round(time.time() - began, 1),
            "modal_region": os.environ.get("MODAL_REGION")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    logs.commit()
    print(f"[xc] {json.dumps(meta)}", flush=True)
    nvml = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                             "--format=csv", "-l", "10"], stdout=open(out / "nvml.csv", "w"),
                            stderr=subprocess.STDOUT)
    rules = StopRules(cfg, began)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out, rules, stop), daemon=True)
    watcher.start()
    env = {**os.environ, "PYTHONPATH": WORKDIR, "YETO_SPOT_RECLAIM_MARKER": RECLAIM_MARKER,
           # yeto.cloud.modal_ckpt_store: the Volume store's commit / reload run in this interpreter
           "YETO_RL_STORE_IMPL": "yeto.cloud.modal_ckpt_store:store_from_env",
           "YETO_RL_STORE_MODAL_VOLUME": XC_VOLUME, "YETO_RL_STORE_MODAL_MOUNT": STORE_MOUNT,
           "YETO_MODAL_PYTHON": sys.executable, "YETO_MODAL_SYSPATH": os.pathsep.join(p for p in sys.path if p)}
    Path(RECLAIM_MARKER).unlink(missing_ok=True)
    _mark_on_signal()
    with open(out / "sampler.log", "ab") as log:
        child = subprocess.Popen([sys.executable, "-m", "yeto.rl.engine.engine_sampler", "--spec",
                                  str(out / "spec.json")], cwd=WORKDIR, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        rc = child.wait()  # a signal only writes the marker (never kills the sampler)
    nvml.terminate()
    stop.set()
    watcher.join(timeout=30)
    result = {**meta, "rc": rc, "stop_rule": rules.reason, "version_groups": rules.version_groups,
              "wall_s": round(time.time() - began, 1)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    logs.commit()
    print(f"[xc] sampler exited {json.dumps(result)}", flush=True)
    return rc


@app.local_entrypoint()
def main(tag: str, store: str = STORE, max_backlog_groups: int = 64, stop_version: int = -1, stop_groups: int = 0,
         idle_s: float = 60.0, max_s: float = 3000.0) -> None:
    if not store.startswith(STORE_MOUNT + "/"):
        raise SystemExit(f"--store must lie under {STORE_MOUNT}")
    cfg = {**DEFAULTS, "store": store, "max_backlog_groups": max_backlog_groups, "stop_version": stop_version,
           "stop_groups": stop_groups, "idle_s": idle_s, "max_s": max_s}
    call = sampler.spawn(tag, cfg)
    print(f"spawned sampler {call.object_id}")
