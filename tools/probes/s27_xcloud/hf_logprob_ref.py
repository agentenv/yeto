"""S27 G-ES: HF transformers (CUDA) reference log-probs for the trainer's first-step samples.

The Ascend trainer's first step (two sides on the v0 weights) logged a train/rollout
log-prob abs_diff of 0.448 against SGLang's ``rollout_log_probs``; S27's known
reference is about 0.01. This container (1 x H200, the S27 Nano image) scores the
same samples teacher-forced with the base weights (version 0):

* ``remote``: the checkpoint's own implementation (``trust_remote_code=True``);
* ``builtin``: transformers' own ``nemotron_h`` implementation, when this
  transformers has one (else recorded as missing);
* ``builtin_naive``: the same with the Mamba CUDA kernels off (``use_mamba_kernels``
  False and the module's kernel symbols set to None), i.e. the naive torch path the
  Ascend trainer runs (no mamba kernels on the NPU); calls of ``torch_forward`` /
  ``cuda_kernels_forward`` are counted to show which path ran.

Input ids per sample = ``tokenizer.encode(prompt)`` + the response ids from the payload.
SGLang tokenizes text input with the same HF tokenizer (``encode`` with special tokens;
for this tokenizer that equals ``add_special_tokens=False``, G-EQ3 checked). Per pair
(SGLang vs remote, SGLang vs builtin, remote vs builtin): mean |diff|, signed mean,
and the mean |diff| by response position (first and last 50 positions).

    YETO_ES_APP=yeto-s27-es-nh1 modal run --detach tools/probes/s27_xcloud/hf_logprob_ref.py --tag s27-es-nh1/ref
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
BASE_DIR = "/models/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/bf77c317"
XC_VOLUME, STORE_MOUNT, GPU_NAME = "yeto-s27-xc", "/root/yeto-checkpoint-store", "NVIDIA H200"
# the trainer's first step took buffer positions 0-3 (G-EQ3: offpolicy_gate kept=4)
PAYLOADS = [f"{STORE_MOUNT}/xc/sample-buffer/payloads/modal-0-b5b4de5a-r0-row{i}.bin" for i in range(4)]

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-hf-ref"))
models = modal.Volume.from_name("yeto-fn-models", create_if_missing=False)
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)
logs = modal.Volume.from_name("yeto-s27-es-logs", create_if_missing=True, version=2)
image = (modal.Image.from_registry(IMAGE).env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"]))


def score(model, samples: list[dict], tok) -> list[list[float]]:
    import torch

    out = []
    for s in samples:
        prompt = tok.encode(s["prompt"])
        response = list(s["tokens"])  # prompt_tokens_known False: response ids only
        ids = torch.tensor([prompt + response], device="cuda")
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, len(prompt):, None])[:, 0]
        out.append([float(x) for x in lp.cpu()])
        del logits, lp
    return out


def compare(a: list[list[float]], b: list[list[float]]) -> dict:
    diffs = [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    flat = [d for row in diffs for d in row]
    finite = [d for d in flat if math.isfinite(d)]
    def by_pos(rows, idx):
        vals = [abs(r[i]) for r in rows for i in idx(len(r)) if 0 <= i < len(r) and math.isfinite(r[i])]
        return round(sum(vals) / max(1, len(vals)), 5)
    return {"tokens": len(flat), "nonfinite": len(flat) - len(finite),
            "mean_abs": round(sum(abs(d) for d in finite) / max(1, len(finite)), 5),
            "signed_mean": round(sum(finite) / max(1, len(finite)), 5),
            "first50_by_pos": [by_pos(diffs, lambda n, k=k: [k]) for k in range(50)],
            "last50_by_pos": [by_pos(diffs, lambda n, k=k: [n - 50 + k]) for k in range(50)],
            "per_sample_mean_abs": [round(sum(abs(d) for d in r) / max(1, len(r)), 4) for r in diffs]}


def naive_paths() -> dict:
    """Turn the built-in nemotron_h Mamba kernels off and count which forward runs."""
    import transformers.models.nemotron_h.modeling_nemotron_h as nh

    off = {}
    for name in ("selective_state_update", "mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        if hasattr(nh, name):
            off[name] = getattr(nh, name) is not None
            setattr(nh, name, None)
    if hasattr(nh, "is_fast_path_available"):
        off["is_fast_path_available"] = bool(nh.is_fast_path_available)
        nh.is_fast_path_available = False
    calls = {"torch_forward": 0, "cuda_kernels_forward": 0}
    mixer = getattr(nh, "NemotronHMamba2Mixer", None)
    for meth in calls if mixer is not None else ():
        if hasattr(mixer, meth):
            orig = getattr(mixer, meth)

            def wrapped(self, *a, _orig=orig, _m=meth, **k):
                calls[_m] += 1
                return _orig(self, *a, **k)
            setattr(mixer, meth, wrapped)
    return {"symbols_were_set": off, "calls": calls, "mixer": mixer is not None}


@app.function(image=image, volumes={"/models": models, STORE_MOUNT: xc, "/data/logs": logs}, gpu="H200:1",
              cpu=8, memory=163840, timeout=2400)
def ref(tag: str, variants: str = "remote,builtin") -> dict:
    sys.path.insert(0, WORKDIR)
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                               capture_output=True, text=True, check=True).stdout.split("\n") if n.strip()]
    if names != [GPU_NAME]:
        raise SystemExit(f"wrong GPUs: {names}")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from yeto.rl.engine.buffer_payload import decode_group

    result: dict = {"gpu": names, "transformers": transformers.__version__, "torch": torch.__version__}
    for mod in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(mod)
            result[mod] = True
        except Exception as error:  # noqa: BLE001 - recorded
            result[mod] = f"missing: {error}"[:200]

    def save() -> None:
        (out / "result.json").write_text(json.dumps(result, indent=1))
        logs.commit()

    samples = [s for p in PAYLOADS for s in decode_group(Path(p).read_bytes())]
    result["samples"] = len(samples)
    result["response_tokens"] = sum(len(s["tokens"]) for s in samples)
    result["weight_versions"] = sorted({(s.get("metadata") or {}).get("weight_version") for s in samples})
    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    result["prompt_tokens"] = [len(tok.encode(s["prompt"])) for s in samples[:4]]
    rollout = [list(s["rollout_log_probs"]) for s in samples]
    scores: dict[str, list[list[float]]] = {"sglang": rollout}
    for label in variants.split(","):
        remote = label == "remote"
        began = time.monotonic()
        try:
            naive = naive_paths() if label == "builtin_naive" else None
            config = AutoConfig.from_pretrained(BASE_DIR, trust_remote_code=remote)
            if naive is not None:
                naive["use_mamba_kernels_was"] = getattr(config, "use_mamba_kernels", "absent")
                config.use_mamba_kernels = False
            model = AutoModelForCausalLM.from_pretrained(BASE_DIR, config=config, trust_remote_code=remote,
                                                         torch_dtype=torch.bfloat16, device_map="cuda")
            model.eval()
            loaded = time.monotonic() - began
            scores[label] = score(model, samples, tok)
            result[label] = {"class": type(model).__name__, "module": type(model).__module__,
                             "load_s": round(loaded, 1), "score_s": round(time.monotonic() - began - loaded, 1),
                             **({"naive": naive} if naive is not None else {})}
            del model
        except Exception as error:  # noqa: BLE001 - recorded; the other implementation still runs
            result[label] = {"error": f"{type(error).__name__}: {error}"[:600]}
        gc.collect()
        torch.cuda.empty_cache()
        save()
    for x, y in (("sglang", "remote"), ("sglang", "builtin"), ("remote", "builtin"),
                 ("sglang", "builtin_naive"), ("builtin", "builtin_naive")):
        if x in scores and y in scores:
            result[f"{x}_vs_{y}"] = compare(scores[x], scores[y])  # diff = x - y
    save()
    (out / "logprobs.json").write_text(json.dumps({k: v for k, v in scores.items()}))
    logs.commit()
    print("[nh] " + json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                if kk not in ("first50_by_pos", "last50_by_pos")}) for k, v in result.items()})[:4000],
          flush=True)
    return result


@app.local_entrypoint()
def main(tag: str, variants: str = "remote,builtin") -> None:
    call = ref.spawn(tag, variants)
    print(f"spawned ref {call.object_id}")
