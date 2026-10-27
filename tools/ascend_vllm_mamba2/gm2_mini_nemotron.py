"""Build a small NemotronH checkpoint from Nano's real weights: first LAYERS layers, EXPERTS routed experts.
usage: python gm2_mini_nemotron.py <nano_dir> <out_dir>   env LAYERS (default 7 -> MEMEM*E), EXPERTS (8), TOPK (4)"""
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    src, dst = sys.argv[1], sys.argv[2]
    L = int(os.environ.get("LAYERS", "7")); E = int(os.environ.get("EXPERTS", "8")); K = int(os.environ.get("TOPK", "4"))
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(f"{src}/config.json"))
    pat = cfg["hybrid_override_pattern"][:L]
    cfg.update(hybrid_override_pattern=pat, num_hidden_layers=L, n_routed_experts=E, num_experts_per_tok=K)
    json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
    for f in os.listdir(src):
        if f.endswith((".py", ".jinja", ".json")) and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f"{src}/{f}", dst)
        if f.startswith("tokenizer") or f == "special_tokens_map.json":
            shutil.copy(f"{src}/{f}", dst)
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    keep = {}
    for k, shard in idx.items():
        if k.startswith("backbone.layers."):
            i = int(k.split(".")[2])
            if i >= L:
                continue
            if ".experts." in k and int(k.split(".experts.")[1].split(".")[0]) >= E:
                continue
        elif k.startswith("mtp"):
            continue
        keep.setdefault(shard, []).append(k)
    out = {}
    for shard, keys in keep.items():
        with safe_open(f"{src}/{shard}", "pt") as fh:
            for k in keys:
                t = fh.get_tensor(k)
                if k.endswith("mixer.gate.weight") or k.endswith("mixer.gate.e_score_correction_bias"):
                    t = t[:E].clone()
                out[k] = t.contiguous()
    save_file(out, f"{dst}/model.safetensors", metadata={"format": "pt"})
    n = sum(t.numel() for t in out.values())
    print("pattern", pat, "experts", E, "topk", K, "tensors", len(out), "params", n, flush=True)

if __name__ == "__main__":
    main()
