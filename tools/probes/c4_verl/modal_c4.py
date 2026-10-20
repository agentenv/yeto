"""T-C4 (S25 N7.6b) GPU check: the verl island's round cuts on a real verl island.

Three island runs of ``yeto.rl.adapters.verl.island_entry`` (Qwen3-0.6B, LoRA,
GSM8K, ``--rl-single-island-no-sync``) in the pinned verl image, one GPU each:

* ``a`` -- commanded by a pre-written trainer-control file (``run 2``, then
  ``stop STOPPED``): trains rounds 0-1, the stop cuts with the SCHEDULED trigger
  into the store (Modal Volume), the island exits;
* ``b`` -- a fresh container: restores the store into a new state dir, resumes
  at rollout 2 (data cursor seek, digest + Adam/scheduler checks), trains 2-3;
* ``c`` -- the continuous control: rounds 0-3, no store (runs beside ``a``).

usage (after the gpu-spend.md pre-registration):
    C4_RUN=<run-id> [C4_GPU=A10G] modal run --detach modal_c4.py
The entrypoint only spawns the chain (a killed detached client would cancel a
blocking ``.remote()``). Evidence: Volume ``t-c4-s25`` under ``/<run>/``.
"""

from __future__ import annotations

import json
import os
import shlex
import sys

import modal

RUN = os.environ.get("C4_RUN", "unset")
YETO = os.environ.get("C4_YETO", "/home/michael/work/s25-c4-cut-ctl")
GPU = os.environ.get("C4_GPU", "A10G")
# Modal A10G reports "NVIDIA A10"
EXPECT = {"A10G": ("NVIDIA A10", "nvidia-a10g"), "L4": ("L4", "nvidia-l4"), "H100!": ("H100", "nvidia-h100")}
sys.path.insert(0, YETO)
from yeto.rl.adapters.verl.image import modal_image  # noqa: E402

vol = modal.Volume.from_name("t-c4-s25", create_if_missing=True)
image = (modal_image(modal)
         .env({"PYTHONPATH": "/yeto", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(f"{YETO}/yeto", "/yeto/yeto", copy=False, ignore=["**/__pycache__"]))
app = modal.App(f"t-c4-{RUN}", image=image)

ROUNDS = 4
PY = "/workspace/verl/.venv/bin/python"  # the pinned verl venv (image.VENV_BIN)
ISLAND_ARGS = [
    "--model", "Qwen/Qwen3-0.6B", "--model-revision", "c1899de289a04d12100db370d81485cdf75e47ca",
    "--data", "zhuzilin/gsm8k", "--data-revision", "0cbd9f31d91ac21a7613dcbc7fef992adac459ae",
    "--rl-single-island-no-sync", "--learner-id", "0", "--reward-function", "gsm8k_reward:score",
    "--global-rounds", str(ROUNDS), "--groups-per-round", "8", "--samples-per-group", "4",
    "--rollout-max-response-len", "256", "--event-tape", "/root/yeto-output/rl-island-0.jsonl",
    "--lora-r", "32", "--lora-targets", "all-linear", "--inner-lr", "1e-5", "--seq-len", "1024",
    "--seed", "17", "--apply-chat-template-kwargs", '{"enable_thinking": false}',
]


def _sh(cmd: str, log) -> int:
    import subprocess

    log.write(f"$ {cmd}\n")
    log.flush()
    return subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT).returncode


@app.function(gpu=GPU, cpu=8.0, memory=49152, timeout=40 * 60, volumes={"/vol": vol})
def island(run: str, phase: str, gpu: str) -> dict:
    import shutil
    import subprocess
    import time

    want_name, group = EXPECT[gpu]
    out = f"/vol/{run}/{phase}"
    os.makedirs(out, exist_ok=True)
    done = f"{out}/result.json"
    if os.path.isfile(done):  # Modal restarts an input after a preemption: never rerun a finished phase
        previous = json.load(open(done))
        if previous.get("rc") == 0:
            return {**previous, "reused_after_restart": True}
    log = open(f"{out}/container.log", "a")
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    _sh(f"nvidia-smi > {out}/nvidia-smi.txt", log)
    if want_name not in name:
        log.write(f"GPU ASSERT FAILED: {name!r} is not {want_name}\n")
        log.close()
        vol.commit()
        return {"phase": phase, "rc": 3, "gpu": name}
    env = dict(os.environ, YETO_RL_COMPAT_GROUP=group, RAY_ADDRESS="127.0.0.1:6379")
    store = f"/vol/{run}/store"
    args = list(ISLAND_ARGS)
    if phase in ("a", "b"):
        args += ["--rl-resume-store", store, "--rl-resume-state-dir", f"/root/state-{phase}"]
    if phase == "a":  # the head's commands for incarnation 0, written through yeto's own writer
        rc = _sh(f"{PY} -c " + shlex.quote(
            "import sys; from yeto.rl.engine import trainer_remote as t; "
            "from yeto.rl.engine.resume import store_for; from yeto.rl.elastic_client import LeaveReason; "
            "w = t.CommandWriter(t.LocalStoreFiles(store_for(sys.argv[1])), 0); "
            "w.send(t.RUN, rounds=2); w.send(t.STOP, reason=LeaveReason.STOPPED); "
            "print(t.island_env(0, None, poll_s=2.0))") + f" {store}", log)
        if rc != 0:
            log.close()
            vol.commit()
            return {"phase": phase, "rc": 4, "gpu": name, "error": "command file not written"}
        env.update({"YETO_TRAINER_CONTROL": "1", "YETO_TRAINER_INCARNATION": "0",
                    "YETO_TRAINER_RESUME": "", "YETO_TRAINER_COMMAND_POLL_S": "2.0"})
    _sh("ray start --head --node-ip-address=127.0.0.1 --port=6379 --num-cpus=32 --num-gpus=1 "
        "--include-dashboard=false --temp-dir=/root/miles-ray", log)
    t0 = time.time()
    with open(f"{out}/island.log", "w") as ilog:
        rc = subprocess.run([PY, "-m", "yeto.rl.adapters.verl.island_entry", *args], env=env,
                            stdout=ilog, stderr=subprocess.STDOUT, cwd="/root").returncode
    seconds = time.time() - t0
    _sh("ray stop --force", log)
    if os.path.isdir("/root/yeto-output"):
        shutil.copytree("/root/yeto-output", f"{out}/yeto-output", dirs_exist_ok=True)
    if os.path.isdir(f"/root/state-{phase}"):
        shutil.copytree(f"/root/state-{phase}", f"{out}/state-dir", dirs_exist_ok=True)
    result = {"phase": phase, "rc": rc, "gpu": name, "island_seconds": round(seconds, 1), "args": args}
    with open(f"{out}/result.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log.close()
    vol.commit()
    return result


@app.function(timeout=90 * 60, volumes={"/vol": vol})
def chain(run: str, gpu: str) -> dict:
    control = island.spawn(run, "c", gpu)
    results = {"a": island.remote(run, "a", gpu)}
    vol.reload()
    if results["a"]["rc"] == 0:
        results["b"] = island.remote(run, "b", gpu)
    results["c"] = control.get()
    vol.reload()
    with open(f"/vol/{run}/chain.json", "w") as fh:
        json.dump(results, fh, indent=1)
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    assert RUN != "unset", "set C4_RUN"
    call = chain.spawn(RUN, GPU)
    print("spawned", RUN, call.object_id, flush=True)
