"""S26 E6 step 3a: the rollout quant gate on one Modal ``H100!`` (Qwen3-4B-Instruct-2507).

One container in the Miles ports image (SGLang 2fa88018) runs
``python -m yeto.rl.quant.gate collect`` for one tier:

* the quantized engine section is the production one: a BF16 sampler spec goes
  through ``sampler_engines.quant_tier.apply`` (tier 1: SGLang online FP8 flags,
  FP8 KV cache and the ignore list; tier 2: the base converted to the FP8 W8A8
  channel checkpoint in the container);
* the BF16 engine section is the same spec without the tier.

Prompts: the first ``prompts`` GSM8K train questions, chat template with
thinking off. Raw per-token data (``report-raw.json``), the report, the engine
logs and NVML go to the results Volume ``<run>/`` and are pulled to
``--out-dir``. The container asserts every GPU is an H100 before anything else.

Run (client venv with modal; the launcher stays on this machine)::

    python tools/probes/s26_e6/gate_modal.py --run e6-gate-t1 --tier 1 --out-dir DIR
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
WORKDIR = "/root/sky_workdir"
RESULTS = "/mnt/results"
RESULTS_VOLUME = "yeto-eval-results"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATA = "zhuzilin/gsm8k"
DATA_REV = "0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
GPU = "H100!"
GPU_NAME = "H100"
# GOAL / gpu-review E6 §3a suggested criteria (not loosened); speed is recorded only
CRITERIA = {"tis_upper": 2.0, "max_mean_abs_diff": 0.05, "max_p99_abs_diff": 0.5,
            "max_tis_clip_share": 0.01, "max_unaligned_share": 0.05}
IGNORE = [".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs", "syncer/target",
          "evidence*", "openspec", "docs", "tutorial", "data"]


def gate(run: str, cfg: dict) -> dict:
    """In the container: GPU check, prompts, engine sections, ``gate collect``."""
    import modal

    sys.path.insert(0, WORKDIR)
    out = Path(RESULTS) / run
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"run": run, "cfg": cfg, "container": os.environ.get("MODAL_TASK_ID"),
                            "start_unix": time.time()}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout.split("\n")
    names = [n.strip() for n in smi if n.strip()]
    meta["gpu"] = names
    (out / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    nvml = subprocess.Popen(["bash", "-c", "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,"
                             f"power.draw --format=csv -l 2 >> {out / 'nvml.csv'}"])
    try:
        if not names or not all(GPU_NAME in n for n in names):
            raise RuntimeError(f"asked for {GPU_NAME}, got {names}")
        import pandas as pd
        import sglang
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer

        from yeto.cloud.modal_eval_island import _image_paths
        from yeto.rl.adapters.sampler_engines import quant_tier

        _image_paths()
        base = snapshot_download(MODEL, revision=MODEL_REV, local_dir="/tmp/qwen3-4b")
        tok = AutoTokenizer.from_pretrained(base)
        frame = pd.read_parquet(hf_hub_download(DATA, "train.parquet", repo_type="dataset", revision=DATA_REV))
        prompts = out / "prompts.jsonl"
        with prompts.open("w") as handle:
            for _, row in frame.head(cfg["prompts"]).iterrows():
                messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                handle.write(json.dumps({"prompt": tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)}) + "\n")
        server = {"base_dir": base, "card_model": names[0], "device_family": "cuda",
                  "engine_version": sglang.__version__, "precision": "bf16", "quant_scheme": "bf16",
                  "topology": "tp1", "max_model_len": 2048, "port": 30000, "python": sys.executable,
                  "start_timeout_s": 1200.0, "in_place_reload": True,
                  "extra_args": ["--mem-fraction-static", "0.8"]}
        bf16 = {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                "server": server, "options": {"tp": 1, "served_model_name": "qwen3-4b"}}
        spec = {"engine": bf16, "base_dir": base, "checks": {},
                "rollout_quant": quant_tier.spec_section(cfg["tier"])}
        began = time.time()
        quant = quant_tier.apply(spec, work_dir=Path("/tmp/e6-quant"))
        meta["prepare_s"] = round(time.time() - began, 1)
        if cfg.get("variant") == "miles-native-block":
            # Miles' own FP8 path: every decoder linear (attention + FFN) block-quantized
            # 128x128 by Miles' converter (the format quantize_params_fp8 sends)
            native = Path("/tmp/e6-quant/miles-fp8-block")
            subprocess.run([sys.executable, "/root/miles/tools/convert_hf_to_fp8.py", "--model-dir", base,
                            "--save-dir", str(native), "--strategy", "block", "--block-size", "128", "128"],
                           check=True, cwd="/root/miles", stdout=subprocess.DEVNULL)
            from yeto.rl.engine.sampler_checks import release_quant_format  # noqa: PLC0415

            _method, fmt = release_quant_format(native)
            (out / "quant-config.json").write_text((native / "config.json").read_text())
            quant = {"engine": {**bf16, "server": {**server, "base_dir": str(native), "precision": "fp8",
                                                    "quant_scheme": "fp8-block-native", "expected_format": fmt}},
                     "rollout_quant": {"variant": "miles-native-block"}}
            meta["rollout_quant"] = quant["rollout_quant"]
        elif cfg.get("variant") == "sglang-online-all":  # SGLang online FP8 without an ignore list
            quant = {"engine": {**bf16, "server": {**server, "precision": "fp8",
                                                    "extra_args": [*server["extra_args"], "--quantization", "fp8"]}},
                     "rollout_quant": {"variant": "sglang-online-all"}}
            meta["rollout_quant"] = quant["rollout_quant"]
        if cfg.get("bf16_kv"):  # diagnosis only: tier 1 without its FP8 KV cache
            args = list(quant["engine"]["server"]["extra_args"])
            i = args.index("--kv-cache-dtype")
            quant["engine"]["server"]["extra_args"] = args[:i] + args[i + 2:]
        meta["rollout_quant"] = quant["rollout_quant"]
        qconfig = Path(quant["engine"]["server"]["base_dir"]) / "config.json"
        (out / "quant-config.json").write_text(qconfig.read_text())
        engines = {"quant": {**quant["engine"], "server": {**quant["engine"]["server"], "port": 30001}},
                   "bf16": bf16}
        (out / "engines.json").write_text(json.dumps(engines, indent=1))
        (out / "criteria.json").write_text(json.dumps(CRITERIA, indent=1))
        argv = [sys.executable, "-m", "yeto.rl.quant.gate", "collect", "--engines", str(out / "engines.json"),
                "--prompts", str(prompts), "--max-tokens", str(cfg["max_tokens"]), "--temperature", "1.0",
                "--n", str(cfg["n"]), "--seed", "1", "--criteria", str(out / "criteria.json"),
                "--out", str(out / "report.json"), "--label", f"tier0->tier{cfg['tier']} {MODEL}@{MODEL_REV[:8]}"]
        meta["argv"] = argv
        with (out / "gate.log").open("w") as log:
            meta["rc"] = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=WORKDIR,
                                        env={**os.environ, "PYTHONPATH": WORKDIR},
                                        timeout=cfg["timeout_s"] - 300).returncode
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        nvml.terminate()
        meta["end_unix"] = time.time()
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        modal.Volume.from_name(RESULTS_VOLUME).commit()
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--tier", type=int, choices=(0, 1, 2), required=True,
                   help="0 = control: BF16 generation scored by BF16 (the floor of the measurement)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout-s", type=int, default=3000)
    p.add_argument("--variant", choices=("miles-native-block", "sglang-online-all"), default=None,
                   help="framework-native FP8 instead of the tier (with --tier 0)")
    p.add_argument("--bf16-kv", action="store_true", help="diagnosis: drop the tier 1 FP8 KV cache flag")
    p.add_argument("--sglang-cast-overlay", action="store_true",
                   help="patch the SGLang fork's FP8 activation cast in the image (sglang_fp8_cast_overlay.py)")
    a = p.parse_args()
    import modal

    sys.path.insert(0, str(REPO))
    from modal._vendor import cloudpickle

    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cfg = {"tier": a.tier, "prompts": a.prompts, "n": a.n, "max_tokens": a.max_tokens, "timeout_s": a.timeout_s,
           "sglang_cast_overlay": a.sglang_cast_overlay, "bf16_kv": a.bf16_kv, "variant": a.variant}
    image = (modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"}))
    if a.sglang_cast_overlay:  # sglang_fp8_cast_overlay.py (the fork casts BF16 activations to FP8)
        overlay = Path(__file__).with_name("sglang_fp8_cast_overlay.py")
        image = (image.add_local_file(str(overlay), "/opt/e6/overlay.py", copy=True)
                 .run_commands("/opt/sglang/bin/python /opt/e6/overlay.py /sgl-workspace/sglang/python"))
    image = image.add_local_dir(str(REPO), WORKDIR, copy=False, ignore=IGNORE)
    app = modal.App(f"yeto-{a.run}")
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
    fn = app.function(image=image, gpu=GPU, cpu=8.0, memory=65536, timeout=a.timeout_s, retries=0,
                      volumes={RESULTS: results}, name="gate", serialized=True)(gate)
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
    pulled = subprocess.run(["modal", "volume", "get", "--force", RESULTS_VOLUME, a.run, str(a.out_dir)],
                            capture_output=True, text=True, timeout=1800)
    result["pull_rc"] = pulled.returncode
    (a.out_dir / "driver.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, default=str)[:2000])
    return 0 if "ok" in result else 1


if __name__ == "__main__":
    sys.exit(main())
