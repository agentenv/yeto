"""S26 E6 step 3b (verl): tier 0 against tier 1 RL on one colocated verl island, Modal ``H100!``.

One container in the verl image (fork acad9875, vLLM 0.29) per run, as the
T-EVAL / E0.1 yeto+verl arm: the GPU type is asserted first (``modal_verl._gpu``),
NVML is sampled every 5 s, a local Ray head starts, and
``yeto.rl.adapters.verl.island_entry --rl-single-island-no-sync`` runs Qwen3-4B-Instruct-2507
with LoRA r=32 on GSM8K (``rounds`` x ``groups`` x ``samples``; TIS token-level, upper 2.0).
Tier 1 adds ``--rl-rollout-quant-tier 1`` (vLLM online FP8 with yeto's ignore list
and FP8 KV cache, ``yeto.rl.quant.tiers.vllm_online_overrides``); nothing else changes.

The island log (verl metrics per step), the event tape and NVML go to the results
Volume ``<run>/island/`` and are pulled to ``--out-dir``.

Run (client venv with modal; the launcher stays on this machine)::

    python tools/probes/s26_e6/rl_verl_modal.py --run e6-t1-verl-a --tier 1 --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.eval import modal_verl as mv  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
GPU = "H100!"


def island(run: str, cfg: dict) -> dict:
    """In the container: one verl island run (module doc)."""
    sys.path.insert(0, mv.WORKDIR)
    out = Path(mv.RESULTS) / run / "island"
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"run": run, "cfg": cfg, "container": os.environ.get("MODAL_TASK_ID")}
    meta["gpu"] = mv._gpu(out, {"gpu": GPU})
    nvml = mv._nvml(out / "nvml.csv")
    began = time.time()
    meta["start_unix"] = began
    tape = out / "island-tape.jsonl"
    try:
        from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec

        spec = AlgorithmSpec(correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
        Path("/tmp/algo.json").write_text(json.dumps(spec.to_dict()))
        env = mv._island_env({"gpu": GPU}, 0)
        argv = [mv.VERL_PY, "-m", "yeto.rl.adapters.verl.island_entry", "--model", MODEL,
                "--model-revision", MODEL_REV, "--data", mv.DATA, "--data-revision", mv.DATA_REV,
                "--learner-id", "0", "--reward-function", "gsm8k_reward:score",
                "--global-rounds", str(cfg["rounds"]), "--groups-per-round", str(cfg["groups"]),
                "--samples-per-group", str(cfg["samples"]), "--rollout-max-response-len", str(cfg["max_tokens"]),
                "--event-tape", str(tape), "--inner-lr", str(cfg["lr"]), "--lora-r", "32",
                "--lora-targets", "all-linear", "--seq-len", str(cfg["seq_len"]), "--seed", str(cfg["seed"]),
                "--rl-algorithm-spec", "/tmp/algo.json", "--rl-single-island-no-sync",
                "--apply-chat-template-kwargs", '{"enable_thinking": false}']
        if cfg["tier"]:
            argv += ["--rl-rollout-quant-tier", str(cfg["tier"])]
        meta["argv"] = argv
        with (out / "island.log").open("w") as log:
            mv._ray_start(env, log)
            meta["rc"] = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/workspace/verl",
                                        timeout=cfg["timeout_s"] - 300).returncode
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        nvml.terminate()
        mv._finish(out, meta, began, {"yeto-output": Path("~/yeto-output").expanduser()})
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--tier", type=int, choices=(0, 1), required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--timeout-s", type=int, default=3600)
    a = p.parse_args()
    import modal
    from modal._vendor import cloudpickle

    from yeto.rl.adapters.verl.image import modal_image

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(mv)
    cfg = {"tier": a.tier, "rounds": a.rounds, "groups": a.groups, "samples": a.samples,
           "max_tokens": a.max_tokens, "seq_len": a.seq_len, "lr": a.lr, "seed": a.seed, "timeout_s": a.timeout_s}
    ignore = [".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs", "syncer/target",
              "evidence*", "openspec", "docs", "tutorial", "data"]
    image = (modal_image(modal).env({"PYTHONUNBUFFERED": "1"})
             .add_local_dir(str(REPO), mv.WORKDIR, copy=False, ignore=ignore))
    app = modal.App(f"yeto-{a.run}")
    results = modal.Volume.from_name(mv.RESULTS_VOLUME, create_if_missing=True, version=2)
    fn = app.function(image=image, gpu=GPU, cpu=8.0, memory=65536, timeout=a.timeout_s, retries=0,
                      volumes={mv.RESULTS: results}, single_use_containers=True, name="island",
                      serialized=True)(island)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    began = time.time()
    with modal.enable_output(), app.run(detach=True):  # a killed client never cancels a spawned call
        (a.out_dir / "app_id.txt").write_text(f"{app.app_id}\n")
        call = fn.spawn(a.run, cfg)
        try:
            result = {"ok": call.get(timeout=a.timeout_s + 600)}
        except Exception as exc:  # noqa: BLE001 - recorded; the raw data is pulled anyway
            result = {"error": f"{type(exc).__name__}: {exc}"}
    result["wall_s"] = round(time.time() - began, 1)
    pulled = subprocess.run(["modal", "volume", "get", "--force", mv.RESULTS_VOLUME, a.run, str(a.out_dir)],
                            capture_output=True, text=True, timeout=1800)
    result["pull_rc"] = pulled.returncode
    (a.out_dir / "driver.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, default=str)[:2000])
    return 0 if "ok" in result else 1


if __name__ == "__main__":
    sys.exit(main())
