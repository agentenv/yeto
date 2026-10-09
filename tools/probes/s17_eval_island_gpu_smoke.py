"""(codex-env.json in OUT: yeto.launcher.codex_container_env(codex_bundle_contract(args, <bundle>/codex)) for qwen35_08b / 4096 / xhigh.)
S17 N11 eval-island GPU smoke: real model (Qwen3.5-0.8B, codex qwen35_08b) on TB2 hold-out 3 tasks.

seed (Modal CPU, same image): version 0 into Volume through the training export path
island (Modal H100!:1): SGLang + Miles session server + codex worker -> TB2 sandboxes -> judge.
Run with reward-env-venv python (modal 1.6.1). Raw output stays in this directory.
"""
import json, os, sys, time, tomllib, threading
from pathlib import Path

REPO = Path("/home/michael/work/s17-eval-infer")
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("N11_OUT", "s1-runs/s17-n11-evalinfer")).resolve(); OUT.mkdir(parents=True, exist_ok=True)
APP = "yeto-s17-n11-evalinfer"
VOLUME = os.environ.get("N11_VOLUME", "yeto-eval-store-n11")
TASKS = ["prove-plus-comm", "constraints-scheduling", "largest-eigenval"]
MODEL, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
HARD_S = int(os.environ.get("N11_HARD_S", "2400"))
SAMPLING = {"rollout_temperature": 1.0, "rollout_top_p": 1.0, "rollout_top_k": None,
            "rollout_max_response_len": 4096, "rollout_max_context_len": 8192,
            "yeto_codex_reasoning_effort": "xhigh", "yeto_codex_max_turns": "12"}

import modal
from yeto.cloud import modal_eval_island as mei

log = open(OUT / "smoke.log", "a")
def say(*a):
    line = " ".join(str(x) for x in a)
    print(time.strftime("%H:%M:%SZ", time.gmtime()), line, file=log, flush=True)
    print(line, flush=True)

prof = tomllib.loads(Path("~/.modal.toml").expanduser().read_text())
prof = next(v for v in prof.values() if isinstance(v, dict) and v.get("active")) if any(
    isinstance(v, dict) and v.get("active") for v in prof.values()) else next(iter(prof.values()))
envs = json.loads((OUT / "codex-env.json").read_text())
envs.update({"SECRLENV_MAX_TURNS": "12", "YETO_HARNESS_TB2_LEASE_SECONDS": "600",
             "YETO_CODEX_WORKER_STDERR": "1",
             "MODAL_TOKEN_ID": prof["token_id"], "MODAL_TOKEN_SECRET": prof["token_secret"]})
spec = mei.EvalFunctionSpec(app_name=APP, volume=VOLUME, workdir=str(REPO),
                            codex_dir="/home/michael/work/codex-bundle/codex", tb2_dir="/home/michael/work/tb2-data",
                            gpu="H100!", gpus=1, cpu=8.0, memory_mib=65536, timeout_s=HARD_S, envs=envs)
app, island_fn, seed_fn = mei.build_eval_app(spec)

config = {
    "store": mei.EVAL_STORE_MOUNT, "volume": VOLUME, "island_id": "eval-n11",
    "holdout": f"{mei.WORKDIR_MOUNT}/data/eval/tb2-holdout.json",
    "eval_data": f"{mei.WORKDIR_MOUNT}/data/eval/tb2-holdout-eval.jsonl",
    "task_ids": TASKS, "trials_v0": int(os.environ.get("N11_TRIALS", "1")), "trials": 1, "assert_gpu_name": "H100",
    "loader": "yeto.rl.eval.codex_infer:loader_factory", "attempt": "yeto.rl.eval.codex_infer:attempt_factory",
    "infer": {"base_model": MODEL, "revision": REV, "tito_model": "qwen35", "context_length": 8192,
              "max_tokens": 4096, "mem_fraction_static": 0.8},
}
events = []
def emit(event, **f):
    events.append({"event": event, "ts": time.time(), **f})
    with open(OUT / "launcher-events.jsonl", "a") as fh:
        fh.write(json.dumps(events[-1], default=str) + "\n")

def watchdog():
    time.sleep(HARD_S + 300)
    say("WATCHDOG: hard limit passed, stopping app")
    os.system(f"/home/michael/work/reward-env-venv/bin/modal app stop {APP} >> {OUT}/watchdog.out 2>&1")
    os._exit(3)
threading.Thread(target=watchdog, daemon=True).start()

say("start", APP, "hard", HARD_S)
with modal.enable_output():
    with app.run():
        say("app run", app.app_id)
        (OUT / "app_id.txt").write_text(str(app.app_id) + "\n")
        t = time.time()
        seed = seed_fn.remote({"volume": VOLUME, "store": mei.EVAL_STORE_MOUNT, "model": MODEL, "revision": REV,
                               "rank": 16, "targets": "attention", "sampling": SAMPLING, "seed": 0,
                               "training_finished": True})
        say("seed done", round(time.time() - t, 1), "s", json.dumps(seed, default=str)[:600])
        (OUT / "seed.json").write_text(json.dumps(seed, indent=1, default=str))
        launcher = mei.EvalIslandLauncher(client=mei.ModalFunctionClient(island_fn), emit=emit, config=config,
                                          gpu="H100!", gpus=1, price_per_hour=None, max_starts=1)
        try:
            summary = launcher.run()
            say("island done", json.dumps(summary, default=str)[:2000])
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        except Exception as exc:  # noqa: BLE001
            say("island FAILED", type(exc).__name__, str(exc)[:2000])
say("app exited")
