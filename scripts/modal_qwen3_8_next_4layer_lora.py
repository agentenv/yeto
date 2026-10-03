"""Run the Qwen3.8-Flash-Next-4layer native LoRA GRPO steps (M4 G0-G4) on Modal H100:4.

    modal run scripts/modal_qwen3_8_next_4layer_lora.py --step g0|g1|g2|g3|g4|g4r|shell
        [--timeout SECONDS] [--cmd 'override bash command'] [--run-id ID]

Each step is one container: the pinned ports image (digest below, pulled with the
ghcr credentials from ~/.docker/config.json as the registry secret, never exported
into the task), yeto's `yeto/` + `scripts/` mounted at /root/yeto, and one Volume
holding HF weights / torch_dist / datasets / checkpoints / logs / compile caches
so steps survive container turnover.  g0 and g1 are CPU-only (image checks,
downloads); g2-g4r reserve H100:4 (`M4_GPU` overrides) with >=256 GiB RAM.  Every
step runs under `timeout` (the launcher's own EXIT trap stops the ray job), the
container's function timeout is a second ceiling, and `modal app stop <app>` is
the external watchdog.  The GPU actually granted is logged at the top of every
step log (/vol/logs/<step>.log) because Modal may upgrade H100 -> H200.
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import time

import modal

IMAGE = (
    "ghcr.io/michaellchung/yeto-miles-ports:c35702e-4e4148f"
    "@sha256:37ac689e29caeecf9faf8587a3ad58c154ecffc7798711d5bd59d792d002b9f9"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.environ.get("M4_APP", "m4-q38n-" + time.strftime("%Y%m%d"))
GPU = os.environ.get("M4_GPU", "H100:4")
NUM_GPUS = int(os.environ.get("YETO_Q38N_NUM_GPUS_PER_NODE", "4"))
VOLUME = os.environ.get("M4_VOLUME", "m4-q38n-vol")
YETO = "/root/yeto"
LINKS = {  # container path -> volume path (Miles' defaults, see yeto.rl.profiles.qwen3_8_next)
    "/root/models": "/vol/models", "/root/ckpt": "/vol/ckpt", "/root/datasets": "/vol/datasets",
    "/root/shared_data": "/vol/shared_data", "/tmp/triton_cache": "/vol/triton_cache",
    "/tmp/inductor_cache": "/vol/inductor_cache", "/root/miles/tensorboard_log": "/vol/tensorboard_log",
}


def _registry_secret() -> modal.Secret | None:
    if not modal.is_local():  # the module is re-imported inside the container; the image is already built
        return None
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = base64.b64decode(auth).decode().split(":", 1)
    return modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})


image = (
    modal.Image.from_registry(IMAGE, secret=_registry_secret()).entrypoint([])
    .add_local_dir(os.path.join(ROOT, "yeto"), f"{YETO}/yeto", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(os.path.join(ROOT, "scripts"), f"{YETO}/scripts", copy=False)
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP, image=image)

LAUNCH = f"bash {YETO}/scripts/run_qwen3_8_next_4layer_lora.sh --yeto-root {YETO}"
COMMON_ENV = f"export YETO_Q38N_NUM_GPUS_PER_NODE={NUM_GPUS} PYTHONPATH={YETO}:/root/miles:/root/Megatron-LM; "
EVAL = ("--eval-interval 5 --eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 2 --eval-max-response-len 512")
STEPS: dict[str, tuple[str, int, bool]] = {  # name -> (command, default timeout, needs gpu)
    "g0": (
        "nproc; free -g | head -2; df -h /dev/shm /tmp /vol | tail -3; hf --version; "
        "git -C /root/miles rev-parse HEAD; "
        "grep -l hc_head_contraction /root/Megatron-LM/megatron/core/transformer/transformer_block.py; "
        "python3 -c 'import miles_plugins.models.qwen3_8_next.lora; print(\"lora plugin import ok\")'; "
        f"{LAUNCH} --dry-run",
        600, False),
    "g1": (f"{LAUNCH} --dry-run >/dev/null && cd {YETO} && python3 -m yeto.rl.profiles.qwen3_8_next download-commands "
           "| while IFS= read -r l; do echo \"+ $l\"; bash -c \"$l\"; done; "
           "python3 -c 'import json;print(\"layers\", json.load(open(\"/root/models/Qwen3.8-Flash-Next-4layer/config.json\"))[\"text_config\"][\"num_hidden_layers\"])'; "
           "ls /root/models/Qwen3.8-Flash-Next-4layer/*.safetensors | wc -l; "
           "ls -la /root/datasets/aime-2024/ /root/datasets/dapo-math-17k/",
           1500, False),
    "g2": (f"bash {YETO}/scripts/convert_qwen3_8_next.sh --variant 4layer --yeto-root {YETO}; "
           "cat /root/ckpt/qwen3.8-flash-next-4layer_torch_dist/latest_checkpointed_iteration.txt; "
           "cat /root/ckpt/qwen3.8-flash-next-4layer_torch_dist/yeto-profile-manifest.json",
           1200, True),
    "g3": (f"YETO_Q38N_RUN_ID=$RUN_ID {LAUNCH} --skip-download --skip-convert --timeout 2700 -- "
           "--use-tensorboard --tb-project-name m4-q38n --tb-experiment-name $RUN_ID",
           2700 + 300, True),
    "g4": (f"YETO_Q38N_RUN_ID=$RUN_ID YETO_Q38N_NUM_ROLLOUT=20 {LAUNCH} --skip-download --skip-convert --timeout 3600 -- "
           f"--use-tensorboard --tb-project-name m4-q38n --tb-experiment-name $RUN_ID {EVAL}; "
           "ls /root/shared_data/$RUN_ID/checkpoints/iter_0000010/adapter /root/shared_data/$RUN_ID/checkpoints/iter_0000020/adapter",
           3600 + 300, True),
    # F4: restart with a NEW run id and --lora-adapter-path (never --load on the LoRA dir)
    "g4r": (f"YETO_Q38N_RUN_ID=$RUN_ID-restart YETO_Q38N_NUM_ROLLOUT=1 {LAUNCH} --skip-download --skip-convert --timeout 1500 -- "
            "--use-tensorboard --tb-project-name m4-q38n --tb-experiment-name $RUN_ID-restart "
            "--lora-adapter-path /root/shared_data/$RUN_ID/checkpoints/iter_0000010/adapter",
            1500 + 300, True),
    "shell": ("bash -c \"$M4_SHELL\"", 600, False),
}


def _run(name: str, cmd: str, timeout_s: int, run_id: str) -> dict:
    for src, dst in LINKS.items():
        os.makedirs(dst, exist_ok=True)
        if os.path.islink(src):
            os.unlink(src)
        elif os.path.isdir(src):
            os.rename(src, src + ".orig")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        os.symlink(dst, src)
    os.makedirs("/vol/logs", exist_ok=True)
    log = f"/vol/logs/{name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1",
                         shell=True, capture_output=True, text=True).stdout.strip()
    header = f"# step={name} app={APP} run_id={run_id} start={time.strftime('%FT%TZ', time.gmtime())} timeout={timeout_s}s\n# gpu:\n{gpu}\n"
    print(header, flush=True)
    start = time.time()
    full = (f"set -o pipefail; {COMMON_ENV} export RUN_ID={shlex.quote(run_id)}; "
            f"timeout --signal=TERM --kill-after=90 {timeout_s} bash -c {shlex.quote(cmd)} 2>&1 | tee -a {log}")
    with open(log, "a") as fh:
        fh.write(header)
    rc = subprocess.run(["bash", "-c", full]).returncode
    elapsed = int(time.time() - start)
    trailer = f"# step={name} rc={rc} elapsed={elapsed}s end={time.strftime('%FT%TZ', time.gmtime())}\n"
    with open(log, "a") as fh:
        fh.write(trailer)
    print(trailer, flush=True)
    vol.commit()
    return {"step": name, "rc": rc, "elapsed_s": elapsed, "gpu": gpu, "log": log, "app": APP, "run_id": run_id}


@app.function(cpu=4, memory=16 * 1024, timeout=1800, volumes={"/vol": vol}, retries=0)
def run_cpu(name: str, cmd: str, timeout_s: int, run_id: str) -> dict:
    return _run(name, cmd, timeout_s, run_id)


@app.function(gpu=GPU, cpu=16, memory=256 * 1024, timeout=4200, volumes={"/vol": vol}, retries=0)
def run_gpu(name: str, cmd: str, timeout_s: int, run_id: str) -> dict:
    return _run(name, cmd, timeout_s, run_id)


@app.local_entrypoint()
def main(step: str, timeout: int = 0, cmd: str = "", run_id: str = "", record: str = ""):
    default_cmd, default_timeout, needs_gpu = STEPS[step]
    command = cmd or default_cmd
    if step == "shell":
        command = command.replace("$M4_SHELL", shlex.quote(cmd))
    timeout_s = timeout or default_timeout
    run_id = run_id or os.environ.get("M4_RUN_ID", "m4-q38n-g4")
    fn = run_gpu if needs_gpu else run_cpu
    print(f"# app={APP} step={step} gpu={GPU if needs_gpu else 'none'} timeout={timeout_s}s run_id={run_id}", flush=True)
    res = fn.remote(step, command, timeout_s, run_id)
    res["cost_h"] = round(res["elapsed_s"] / 3600, 3)
    print(json.dumps(res))
    if record:
        with open(record, "a") as fh:
            fh.write(json.dumps(res) + "\n")
    if res["rc"] != 0:
        raise SystemExit(res["rc"])
