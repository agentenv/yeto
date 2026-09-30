# tf-diagnosis: pairwise LoRA gradient comparison (rel L2 = ||a-b||/||a||, cosine), all + per projection type
import sys, json, numpy as np, re, itertools
def load(d, isl):
    p = f"{d}/work/seed-17/yeto-federated-m2/island-{isl}/audit/round-00000001.grad"
    idx = json.load(open(p + ".json")); return idx, np.fromfile(p + ".f32", dtype="<f4").astype(np.float64)
def m(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a), a @ b / np.linalg.norm(a) / np.linalg.norm(b)
runs = sys.argv[1:]
for isl in (0, 1):
    L = {r: load(r, isl) for r in runs}
    idx = L[runs[0]][0]
    print(f"== island {isl}  norms: " + ", ".join(f"{r.split('/')[-1]}={np.linalg.norm(L[r][1]):.6f}" for r in runs))
    for a, b in itertools.combinations(runs, 2):
        ga, gb = L[a][1], L[b][1]
        r, c = m(ga, gb); same = np.array_equal(ga, gb)
        s = f"  {a.split('/')[-1]:>10} vs {b.split('/')[-1]:<10} all: relL2={r:.5f} cos={c:.6f} bitexact={same}"
        groups = {}
        for sp in idx["specs"]:
            k = re.search(r"\.(\w+_proj)\.lora_(A|B)", sp["name"]); key = k.group(1) + "." + k.group(2)
            groups.setdefault(key, []).append(slice(sp["offset"], sp["offset"] + sp["numel"]))
        per = []
        for key in sorted(groups):
            if key.endswith("A"): continue
            x = np.concatenate([ga[s_] for s_ in groups[key]]); y = np.concatenate([gb[s_] for s_ in groups[key]])
            per.append(f"{key[:-2]}={m(x, y)[0]:.4f}")
        # per-tensor distribution
        tr = [m(ga[sp["offset"]:sp["offset"]+sp["numel"]], gb[sp["offset"]:sp["offset"]+sp["numel"]])[0]
              for sp in idx["specs"] if "lora_B" in sp["name"]]
        print(s); print("     perproj relL2 " + " ".join(per)); print(f"     per-tensor relL2 median={np.median(tr):.4f} max={np.max(tr):.4f}")
