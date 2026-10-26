"""S26 E6 probe (Modal H100!, verl image, vLLM 0.29): does vLLM generate sensibly with
the tier 1 verl settings, without verl's weight sync?

e6-t1-verl-a-r2 generated 512-token garbage with NaN log-probs (score 0) on step 1.
This probe starts vLLM ``LLM`` on Qwen3-4B-Instruct-2507 with exactly the
``quantization`` and ``hf_overrides`` yeto passes for tier 1
(``tiers.vllm_online_overrides``), with and without ``enable_lora``, generates 16
GSM8K answers at temperature 0 and records lengths, a text sample and whether
any log-prob is NaN. A BF16 engine is the control.

Run: ``python tools/probes/s26_e6/vllm_fp8_probe.py --out DIR`` (client venv with modal).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.eval import modal_verl as mv  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REV = "cdbee75f17c01a7cc42f958dc650907174af0554"

CONTAINER = r'''
import json, math, sys
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
case = json.loads(sys.argv[1])
base = snapshot_download(case["model"], revision=case["rev"])
kw = dict(model=base, max_model_len=1024, gpu_memory_utilization=0.6, enforce_eager=False)
if case["fp8"]:
    kw.update(quantization="fp8")
    if case.get("hf_overrides"):
        kw.update(hf_overrides=case["hf_overrides"])
if case["lora"]:
    kw.update(enable_lora=True, max_loras=1, max_lora_rank=32)
llm = LLM(**kw)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in case["questions"]]
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512, logprobs=0))
rows = []
for o in outs:
    c = o.outputs[0]
    lps = [list(d.values())[0].logprob for d in (c.logprobs or [])]
    rows.append({"tokens": len(c.token_ids), "nan": sum(1 for x in lps if not math.isfinite(x)), "text": c.text[:200]})
print("RESULT " + json.dumps(rows))
'''


def probe(cases: list[dict]) -> list[dict]:
    out = []
    mv._gpu(Path("/tmp"), {"gpu": "H100!"})
    for case in cases:
        proc = subprocess.run([mv.VERL_PY, "-c", CONTAINER, json.dumps(case)], capture_output=True, text=True,
                              timeout=1500)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        out.append({"name": case["name"], "rc": proc.returncode,
                    "rows": json.loads(line[7:]) if line else None, "stderr_tail": proc.stderr[-3000:]})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--round", type=int, default=1)
    a = p.parse_args()
    import modal
    from modal._vendor import cloudpickle

    from yeto.rl.adapters.verl.image import modal_image

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(mv)
    # the exact value of e6-t1-verl-a-r2 (``tiers.vllm_online_overrides``), saved from its argv
    hf = json.loads(Path(a.out / "hf_overrides.json").read_text())
    questions = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                 "How many clips did Natalia sell altogether in April and May?",
                 "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
                 "How much did she earn?"] * 8
    common = {"model": MODEL, "rev": MODEL_REV, "questions": questions, "hf_overrides": hf}
    verl_native = {"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                           "weight_block_size": [128, 128],
                                           "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(36)]}}
    if a.round == 1:
        cases = [{**common, "name": "bf16", "fp8": False, "lora": False},
                 {**common, "name": "fp8", "fp8": True, "lora": False},
                 {**common, "name": "fp8+lora", "fp8": True, "lora": True}]
    else:  # round 2: verl's own FP8 config (no yeto ignore list) and vLLM online FP8 (no hf_overrides)
        cases = [{**common, "name": "verl-native-config", "fp8": True, "lora": False, "hf_overrides": verl_native},
                 {**common, "name": "vllm-online", "fp8": True, "lora": False, "hf_overrides": None},
                 {**common, "name": "vllm-online+lora", "fp8": True, "lora": True, "hf_overrides": None}]
    image = modal_image(modal).env({"PYTHONUNBUFFERED": "1"})
    app = modal.App("yeto-e6-vllm-fp8-probe")
    fn = app.function(image=image, gpu="H100!", cpu=8.0, memory=65536, timeout=3000, retries=0,
                      name="probe", serialized=True)(probe)
    with modal.enable_output(), app.run(detach=True):
        (a.out / "app_id.txt").write_text(f"{app.app_id}\n")
        result = fn.spawn(cases).get(timeout=3600)
    (a.out / f"result-{a.round}.json").write_text(json.dumps(result, indent=1))
    for r in result:
        print(r["name"], r["rc"], [(x["tokens"], x["nan"]) for x in (r["rows"] or [])][:4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
