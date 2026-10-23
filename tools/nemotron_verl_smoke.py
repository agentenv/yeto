"""GPU smoke: verl Megatron full-parameter GRPO on Nemotron-3 Nano 30B-A3B (H200), MFU.

Usage (local):
  python tools/nemotron_verl_smoke.py prep  RUN_ID          # CPU: image build, weights, data, imports
  python tools/nemotron_verl_smoke.py smoke RUN_ID [N_GPUS] # GPU: a few GRPO steps, spawned detached
  python tools/nemotron_verl_smoke.py pull  RUN_ID          # fetch small result files, stop the app

The run uses yeto's verl image plus :func:`megatron_recipe.image_commands` and
the overrides from :func:`megatron_recipe.build_overrides`; the trainer is the
bare verl ``main_ppo`` (not yeto's island loop or syncer). Weights stay on the
Modal volume; ``pull`` never fetches files over 50 MB.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

APP = "yeto-nemotron-smoke-"
VOLUME = "yeto-nemotron"
MNT = "/mnt/nem"
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
VERL_PY = "/workspace/verl/.venv/bin/python"
MAX_PULL = 50 << 20


def _sh(cmd: str, log: Path, env: dict | None = None, timeout: int = 7200) -> int:
    import os
    with log.open("a") as fh:
        fh.write(f"\n$ {cmd}\n")
        fh.flush()
        p = subprocess.run(["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})}, timeout=timeout)
    return p.returncode


def prep(run: str) -> dict:
    import modal
    out = Path(MNT) / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    log = out / "prep.log"
    model_dir = Path(MNT) / "models" / MODEL_ID.split("/")[1]
    rc = {}
    if not (model_dir / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_dir=str(model_dir), max_workers=16)
    rc["model_files"] = len(list(model_dir.glob("*.safetensors")))
    data = Path(MNT) / "data" / "gsm8k"
    if not (data / "train.parquet").exists():
        rc["gsm8k"] = _sh(f"cd /workspace/verl && {VERL_PY} examples/data_preprocess/gsm8k.py "
                          f"--local_save_dir {data}", log)
    rc["imports"] = _sh(
        f"{VERL_PY} -c 'import mamba_ssm, causal_conv1d, megatron.core, megatron.bridge, vllm, torch;"
        "from megatron.bridge import AutoBridge;"
        f"print(torch.__version__, vllm.__version__, mamba_ssm.__version__, causal_conv1d.__version__);"
        f"print(AutoBridge.can_handle(\"{model_dir}\", trust_remote_code=True))'", log)
    modal.Volume.from_name(VOLUME).commit()
    return rc


def _gpu_names() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout
    return [n.strip() for n in out.splitlines() if n.strip()]


def smoke(run: str, plan: dict) -> dict:
    import modal
    from yeto.rl.adapters.verl.megatron_recipe import NemotronMegatronPlan, build_overrides
    out = Path(MNT) / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    meta = {"began": time.time(), "plan": plan}
    names = _gpu_names()
    meta["gpus"] = names
    if len(names) != plan["n_gpus"] or not all(re.search(r"H200(?![\w-])", n) for n in names):
        meta["error"] = f"asked for {plan['n_gpus']}x H200, got {names}"
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        modal.Volume.from_name(VOLUME).commit()
        raise RuntimeError(meta["error"])
    nvml = subprocess.Popen(["bash", "-c", "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,"
                             f"power.draw --format=csv -l 5 >> {out}/nvml.csv"])
    overrides = build_overrides(NemotronMegatronPlan(**plan))
    (out / "overrides.txt").write_text("\n".join(overrides) + "\n")
    cmd = "cd /workspace/verl && " + VERL_PY + " -m verl.trainer.main_ppo " + " ".join(f"'{o}'" for o in overrides)
    env = {"VERL_FILE_LOGGER_PATH": f"{out}/metrics.jsonl", "VLLM_USE_V1": "1", "RAY_TMPDIR": "/tmp/ray",
           "HYDRA_FULL_ERROR": "1"}
    try:
        meta["rc"] = _sh(cmd, out / "train.log", env, timeout=int(plan.get("_timeout", 5400)))
    finally:
        nvml.terminate()
        meta["seconds"] = round(time.time() - meta["began"], 1)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        modal.Volume.from_name(VOLUME).commit()
    return meta


def build_app(run: str, n_gpus: int):
    import modal
    from modal._vendor import cloudpickle
    from yeto.rl.adapters.verl.image import modal_image
    from yeto.rl.adapters.verl.megatron_recipe import image_commands
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ignore = [".git", ".venv", ".claude", "**/__pycache__", "syncer/target", "evidence*", "openspec", "docs",
              "tutorial", "data", "s1-runs"]
    image = (modal_image(modal, patched=False).run_commands(*image_commands())
             .pip_install("huggingface_hub[hf_transfer]==0.36.0")
             .env({"PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/yeto"})
             .add_local_dir(str(REPO / "yeto"), "/root/yeto/yeto", copy=False))
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
    app = modal.App(APP + run)
    fns = {
        "prep": app.function(image=image, cpu=16.0, memory=65536, timeout=7200, volumes={MNT: vol},
                             ephemeral_disk=600 * 1024, name="prep", serialized=True)(prep),
        "smoke": app.function(image=image, gpu=f"H200!:{n_gpus}", cpu=32.0, memory=786432, timeout=7200,
                              retries=0, volumes={MNT: vol}, ephemeral_disk=1024 * 1024, name="smoke",
                              serialized=True)(smoke),
    }
    return app, fns


def plan_for(n_gpus: int) -> dict:
    root = f"{MNT}"
    return dict(model_path=f"{root}/models/{MODEL_ID.split('/')[1]}", train_file=f"{root}/data/gsm8k/train.parquet",
                val_file=f"{root}/data/gsm8k/test.parquet", out_dir="/tmp/out", n_gpus=n_gpus, tp=n_gpus,
                ep=n_gpus, rollout_tp=n_gpus, groups_per_round=32, samples_per_group=8, max_prompt_len=512,
                max_response_len=1024, total_steps=4, num_experts=128)


def main() -> None:
    import modal
    action, run = sys.argv[1], sys.argv[2]
    n_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    local = REPO / "s1-runs" / "nemotron" / run
    local.mkdir(parents=True, exist_ok=True)
    if action == "pull":
        listing = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "--json", VOLUME, f"runs/{run}"],
                                 capture_output=True, text=True, cwd="/tmp")
        for entry in json.loads(listing.stdout or "[]"):
            size = int(str(entry.get("Size", "0")).split()[0]) if str(entry.get("Size", "0")).split()[0].isdigit() else 0
            name = entry.get("Filename") or entry.get("path")
            if entry.get("Type") == "file" and size < MAX_PULL:
                subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOLUME, name,
                                str(local)], cwd="/tmp")
        app_id = (local / "app_id.txt").read_text().strip() if (local / "app_id.txt").exists() else None
        if app_id:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", app_id], cwd="/tmp")
        return
    app, fns = build_app(run, n_gpus)
    with modal.enable_output(), app.run(detach=True):
        (local / "app_id.txt").write_text(app.app_id + "\n")
        call = fns[action].spawn(run) if action == "prep" else fns["smoke"].spawn(run, plan_for(n_gpus))
        (local / "call_id.txt").write_text(call.object_id + "\n")
        print("spawned", action, call.object_id, app.app_id, flush=True)


if __name__ == "__main__":
    main()
