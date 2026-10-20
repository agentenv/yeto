"""Miles cells: the yeto launcher on Modal through ``miles_local.sh`` / ``miles_head.sh``.

:func:`launcher_args` builds the launcher argument strings of a cell (the store
is ``modal-volume://<STORE_VOLUME>/<run id>``); :func:`run` writes them into the
script environment and runs the script, which pulls everything into ``raw/``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = "/home/michael/work/gpu-head/venv/bin/python"  # modal client + launcher deps (S25 C1/C3)
SKY = "/home/michael/work/gpu-head/venv/bin/sky"
STORE_VOLUME = "yeto-eval-store"
TAPE_VOLUME = "yeto-event-tapes"
MODEL = "--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"
DATA = "--data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
SPEC = "--rl-algorithm-spec examples/rl_algorithms/decoupled-ppo-c2.json"  # max_policy_staleness 2
SPEC_AGE4 = "--rl-algorithm-spec examples/rl_algorithms/decoupled-ppo-c2-agentic.json"  # staleness 4 (S25 C1)


def _gpu(cfg: dict, n: int) -> str:
    card = {"A10G": "a10g", "H100!": "h100"}[cfg["gpu"]]
    return ",".join([f"modal:1x{card}"] * n)


def _common(cfg: dict, run_id: str, prefix: str) -> str:
    return (f"launch --training-mode rl --rl-engine ports --modal-gpu-exact --modal-retries 0 --modal-cpu 8 "
            f"--modal-memory 64 --modal-tape-volume {TAPE_VOLUME} --controller local --cluster-prefix {prefix} "
            f"--tuning lora --lora-r {cfg['lora_r']} --rl-observe-timeline {MODEL} {DATA} "
            f"--rl-checkpoint-store modal-volume://{STORE_VOLUME}/{run_id} "
            "--reward-function gsm8k_reward:score --fragments 1 --pipeline 1 "
            f"--rollout-batch-size {cfg['rollout_batch_size']} --n-samples-per-prompt {cfg['n_samples']} "
            f"--rollout-max-response-len {cfg['max_response_len']} --seq-len {cfg['seq_len']} "
            f"--inner-lr {cfg['lr']} --seed {cfg['seed']} "
            "--apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code")


def launcher_args(cell, run_id: str) -> dict[str, str]:
    """The script environment of a local-controller Miles cell (ARGS, SARGS, reclaim)."""
    cfg, cap = cell.config, cell.capability
    if cell.harness != "miles_local":
        raise ValueError(f"{cell.key} is not a local-controller Miles cell")
    common = _common(cfg, run_id, run_id)
    env: dict[str, str] = {"ISLANDS": str(cfg["islands"])}
    if cap == "E1":
        env["ARGS"] = f"{common} --gpu {_gpu(cfg, 1)} --rl-single-island-no-sync --total-steps {cfg['rounds']}"
    elif cap == "E2":  # S25 C1: islands 0..k-1 sample, island k trains
        k = cfg["samplers"]
        roles = " ".join(f"--rl-island-sample-role {i}:sampler" for i in range(k))
        env["ARGS"] = (f"{common} --gpu {_gpu(cfg, k + 1)} --rl-single-island-no-sync {roles} "
                       f"--rl-island-sample-role {k}:trainer {SPEC_AGE4} --rl-max-policy-age {cfg['max_policy_age']} "
                       "--rl-sampler-max-backlog-rounds 2 --rl-sampler-min-round-seconds 60 --rl-sample-wait-s 900 "
                       f"--total-steps {cfg['rounds']}")
        env.update(RECLAIM_LABEL=cfg["reclaim_label"], RECLAIM_ROUND=str(cfg["reclaim_round"]))
    elif cap == "E4":  # two launches on one store: the sampler runs ahead of the trainer
        age = cfg["max_policy_age"]
        env["ARGS"] = (f"{common} --gpu {_gpu(cfg, 1)} --rl-single-island-no-sync --rl-sample-role trainer "
                       f"{SPEC} --rl-max-policy-age {age} --rl-sample-wait-s 900 --total-steps {cfg['rounds']}")
        env["SARGS"] = (f"{_common(cfg, run_id, run_id + '-s')} --gpu {_gpu(cfg, 1)} --rl-single-island-no-sync "
                        f"--rl-sample-role sampler {SPEC} --rl-max-policy-age {age} "
                        f"--rl-sampler-max-backlog-rounds {cfg['backlog_rounds']} "
                        f"--total-steps {cfg['sampler_rounds']}")
    elif cap == "E5":  # S24 L5.4: a pair whose trainer the head's S1 scheduler stops and restarts
        env["ARGS"] = (f"{common} --gpu {_gpu(cfg, 2)} --rl-single-island-no-sync "
                       "--rl-island-sample-role 0:sampler --rl-island-sample-role 1:trainer "
                       f"{SPEC} --rl-max-policy-age {cfg['max_policy_age']} --rl-cut-every 1 "
                       f"--rl-sampler-max-backlog-rounds 3 --rl-sampler-min-round-seconds {cfg['min_round_s']} "
                       "--rl-sample-wait-s 900 --rl-trainer-scheduler s1 --rl-trainer-island 1 "
                       "--rl-trainer-scheduler-interval 15 --rl-trainer-start-seconds 30 "
                       "--rl-trainer-start-seconds-fixed --rl-trainer-ready-seconds 2700 "
                       f"--rl-trainer-drain-seconds 2400 --total-steps {cfg['rounds']}")
    return env


def run(cell, run_id: str, run_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    raw = run_dir / "raw"
    cfg = cell.config
    env = dict(os.environ, P=run_id, R=str(raw), REPO=str(REPO), PY=PY, SKY=SKY, STOREVOL=STORE_VOLUME,
               TAPEVOL=TAPE_VOLUME, HARD=str(cfg["hard_s"]), HOME="/home/michael")
    if cell.harness == "miles_head":
        script = HERE / "miles_head.sh"
        env.update(GPU={"A10G": "a10g", "H100!": "h100"}[cfg["gpu"]], STEPS=str(cfg["rounds"]))
        if dry_run:
            env["PLAN_ONLY"] = "1"
    else:
        script = HERE / "miles_local.sh"
        env.update(launcher_args(cell, run_id))
        if dry_run:
            env["DRY"] = "1"
    log = run_dir / ("dry-run.log" if dry_run else "driver.log")
    with log.open("w") as fh:
        rc = subprocess.run(["bash", str(script)], env=env, stdout=fh, stderr=subprocess.STDOUT, cwd="/tmp").returncode
    return {"script": script.name, "rc": rc, "log": str(log)}


def card_seconds(raw: Path) -> float:
    """Card seconds from the container list the script polled every 20 s: per
    container, last seen - first seen + 40 s (the S25 C1 estimate)."""
    import datetime

    seen: dict[str, list[float]] = {}
    path = Path(raw) / "containers.log"
    if not path.is_file():
        return 0.0
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        stamp = datetime.datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        for container in parts[1:]:
            seen.setdefault(container, []).append(stamp)
    return sum(max(t) - min(t) + 40.0 for t in seen.values())
