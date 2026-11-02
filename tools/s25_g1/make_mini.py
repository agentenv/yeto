"""Build the S25 G1 rehearsal checkpoints from Miles' 4-layer GLM5.3-Flash CI model.

Runs inside the prep container (python3 make_mini.py ROOT). Writes, under ROOT:

* ``src-4layer/``: CharyZeng/GLM-5.3-Flash-4layer at REVISION (288 experts, bf16);
* ``glm5next-mini-bf16/``: the same model with the first KEEP routed experts per
  MoE layer (router rows and expert bias cut to match; n_routed_experts=KEEP);
* ``glm5next-mini-mxfp4-template/``: layout.json + metadata for the MXFP4 layout
  of yetalabs/apex-flash-1-mxfp4-experts@70e86a0 (routed experts
  ``weight_packed`` U8 + E8M0 ``weight_scale``, everything else as the bf16 source),
  with that export's quantization_config;
* ``glm5next-mini-mxfp4-r0/``: round 0 for the sampler, converted from the bf16
  mini model by the N8.5 converter (``requantize_checkpoint``);
* ``datasets/gsm8k-0cbd9f3/train.jsonl`` and the HF dataset cache under ``hf/``.

Prints one JSON summary line (``MINI-SUMMARY {...}``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = "CharyZeng/GLM-5.3-Flash-4layer"
REVISION = "7fb7883207fe73ea04430ac392f6ffe2b9f936c4"
MXFP4_REPO = "yetalabs/apex-flash-1-mxfp4-experts"
MXFP4_REVISION = "70e86a0"
DATA_REPO = "zhuzilin/gsm8k"
DATA_REVISION = "0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
KEEP = int(os.environ.get("MINI_EXPERTS", "32"))
EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")
ROUTED = re.compile(r"^(.*\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.weight$")


def truncate(src: Path, dst: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map, total, cut = {}, 0, 0
    for shard in sorted(set(index.values())):
        out = {}
        with safe_open(str(src / shard), framework="pt") as f:
            for name in f.keys():
                m = EXPERT.search(name)
                if m and int(m.group(1)) >= KEEP:
                    cut += 1
                    continue
                t = f.get_tensor(name)
                if name.endswith(".mlp.gate.weight") or name.endswith(".mlp.gate.e_score_correction_bias"):
                    t = t[:KEEP].clone()
                out[name] = t.contiguous()
        if out:
            save_file(out, str(dst / shard), metadata={"format": "pt"})
            for name, t in out.items():
                weight_map[name] = shard
                total += t.numel() * t.element_size()
        print(f"[mini] {shard}: kept {len(out)}", flush=True)
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True))
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json" \
                and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
    config = json.loads((dst / "config.json").read_text())
    config["text_config"]["n_routed_experts"] = KEEP
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    return {"tensors": len(weight_map), "bytes": total, "cut_expert_tensors": cut}


def make_template(bf16: Path, template: Path, quant_config: dict) -> dict:
    from yeto.rl.quant.requant import (LAYOUT_FILE, LAYOUT_SCHEMA, CheckpointLayout, TensorSpec,
                                       metadata_files, read_header)
    from yeto.rl.quant.mxfp4 import GROUP_SIZE

    weight_map = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"]
    tensors: dict[str, TensorSpec] = {}
    headers = {s: read_header(bf16 / s) for s in sorted(set(weight_map.values()))}
    for name, shard in weight_map.items():
        info = headers[shard][name]
        m = ROUTED.match(name)
        if m:
            rows, cols = info["shape"]
            tensors[m.group(1) + ".weight_packed"] = TensorSpec("U8", (rows, cols // 2), shard)
            tensors[m.group(1) + ".weight_scale"] = TensorSpec("U8", (rows, cols // GROUP_SIZE), shard)
        else:
            tensors[name] = TensorSpec(info["dtype"], tuple(info["shape"]), shard)
    layout = CheckpointLayout(tensors, quant_config)
    layout.validate()
    template.mkdir(parents=True, exist_ok=True)
    for f in metadata_files(bf16):
        shutil.copy2(f, template / f.name)
    config = json.loads((template / "config.json").read_text())
    config["quantization_config"] = quant_config
    (template / "config.json").write_text(json.dumps(config, indent=2))
    payload = {"schema": LAYOUT_SCHEMA, "quantization_config": quant_config,
               "tensors": {n: {"dtype": t.dtype, "shape": list(t.shape), "shard": t.shard}
                           for n, t in sorted(tensors.items())}}
    (template / LAYOUT_FILE).write_text(json.dumps(payload, sort_keys=True))
    return {"quantized_modules": len(layout.quantized_modules), "tensors": len(tensors)}


def main(root: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    root_dir = Path(root)
    os.environ.setdefault("HF_HOME", str(root_dir / "hf"))
    summary: dict = {"keep_experts": KEEP}
    src = Path(snapshot_download(REPO, revision=REVISION, local_dir=str(root_dir / "src-4layer")))
    bf16 = root_dir / "glm5next-mini-bf16-7fb7883"
    if not (bf16 / "model.safetensors.index.json").is_file():
        summary["truncate"] = truncate(src, bf16)
    mx_config = json.loads(Path(hf_hub_download(MXFP4_REPO, "config.json", revision=MXFP4_REVISION)).read_text())
    quant_config = mx_config["quantization_config"]
    template = root_dir / "glm5next-mini-mxfp4-template"
    summary["template"] = make_template(bf16, template, quant_config)
    from yeto.rl.quant.awq import AwqSmoothing
    from yeto.rl.quant.requant import CheckpointLayout, requantize_checkpoint

    r0 = root_dir / "glm5next-mini-mxfp4-r0-7fb7883"
    if r0.exists():
        shutil.rmtree(r0)
    summary["r0"] = requantize_checkpoint(bf16, template, r0, AwqSmoothing.none())
    check = CheckpointLayout.from_checkpoint(r0)  # the stage-1 template path (no layout.json)
    summary["r0_quantized_modules"] = len(check.quantized_modules)
    data = Path(snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REVISION))
    import pandas as pd

    rows = pd.read_parquet(data / "train.parquet")
    out = root_dir / "datasets/gsm8k-0cbd9f3"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for row in rows.to_dict(orient="records"):
            f.write(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()},
                               default=str) + "\n")
    summary["gsm8k_rows"] = len(rows)
    summary["gsm8k_columns"] = list(rows.columns)
    summary["gsm8k_row0"] = str(rows.iloc[0].to_dict())[:400]
    print("MINI-SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
