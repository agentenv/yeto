"""Derive the 6.2 tolerance header from the legacy repeats (uses scripts/rl_engine_equivalence.py).

loss / grad_norm are None in the legacy rl_local_round events (and loss is None on ports too),
so they are filled from Miles' per-step `train/loss` / `train/grad_norm` log line
(optimizer_steps=1 => Miles step k == round k+1); the source is recorded per value."""
import ast, glob, json, re, sys, time
from pathlib import Path
sys.path.insert(0, "/home/michael/work/gpu-legacy/yeto/scripts")
import rl_engine_equivalence as eq

STEP = re.compile(r"log_utils\.py:\d+ - step (\d+): (\{.*\})")

def miles_steps(log):
    out = {}
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = STEP.search(line)
        if m and "(MegatronTrainRayActor" in line:
            out[int(m.group(1)) + 1] = ast.literal_eval(m.group(2))
    return out

def rounds_for(run_dir):
    arm = Path(run_dir) / "work/seed-17/yeto-federated-m2"
    rounds = eq.extract_rounds(eq.read_events(sorted(arm.glob("island-*/events.jsonl"))))
    src = {}
    for isl in sorted(arm.glob("island-*")):
        i = int(isl.name.split("-")[1]); steps = miles_steps(isl / "miles.log")
        for (ii, r), row in rounds.items():
            if ii != i: continue
            for m, key in (("loss", "train/loss"), ("grad_norm", "train/grad_norm")):
                if row.get(m) is None and r in steps and key in steps[r]:
                    row[m] = float(steps[r][key]); src[f"{i}/{r}/{m}"] = "miles.log"
    return rounds, src

if __name__ == "__main__":
    base = Path(sys.argv[1]); runs = sys.argv[2:]
    reps, srcs = [], {}
    for r in runs:
        rows, src = rounds_for(base / r); reps.append(rows); srcs[r] = src
    tol = eq.derive_tolerance(reps, factor=2.0, floor=1e-6)
    meta = {"mode": "real (legacy side only)", "fake": False, "created_unix": time.time(),
            "config": ("benchmark_rl.py federated/strict-avg, Qwen/Qwen3-0.6B@c1899de, GSM8K (/work/data/gsm8k.jsonl, "
                       "train/eval manifests == ports strict2 run), 2 islands x 1 H100, 3 global rounds, "
                       "4 groups x 8 samples, LoRA r16 all-linear, seed 17, --rl-engine legacy"),
            "legacy_repeats": len(reps), "tolerance_factor": 2.0, "tolerance_floor": 1e-6}
    hdr = eq.header_text(meta, tol)
    per = {r: {f"{k[0]}/{k[1]}": v for k, v in sorted(rows.items())} for r, rows in zip(runs, reps)}
    (base / "report.md").write_text(hdr, encoding="utf-8")
    (base / "tolerance.json").write_text(json.dumps({"meta": meta, "tolerance": tol, "legacy_repeats": per,
                                                     "filled_from_miles_log": srcs}, indent=2, default=str))
    print(hdr); print(json.dumps(per, indent=1, default=str))
