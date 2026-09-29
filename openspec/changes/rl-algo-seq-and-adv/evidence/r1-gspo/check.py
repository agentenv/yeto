"""Pre-declared R1 criteria (plan.md). usage: check.py <run dir>"""
import glob, json, math, re, sys
d = sys.argv[1]
rc = open(f"{d}/rc").read().strip()
log = open(f"{d}/launch.log", errors="replace").read()
ev = [json.loads(l) for f in glob.glob(f"{d}/events/*.jsonl") for l in open(f) if l.strip()]
rounds = sorted((e for e in ev if e.get("event") == "rl_round_trained"), key=lambda e: e["rollout_id"])
# Miles per-step metrics, deduplicated (log_utils and model.py print the same dict)
steps, seen = [], set()
for m in re.finditer(r"log_utils\.py:\d+ - step (\d+): \{[^}]*'train/pg_clipfrac': ([^,}]+)", log):
    key = m.start()
    steps.append((int(m.group(1)), float(m.group(2))))
per_round, cur = [], []
for step, v in steps:
    if step == 0 and cur:
        per_round.append(cur); cur = []
    cur.append(v)
if cur:
    per_round.append(cur)
checks = {"exit_code": rc, "three_rounds": len(rounds) == 3 and len(per_round) == 3,
          "finalized": any(e.get("event") == "rl_learner_finalized" for e in ev)}
rows = []
for r, vals in zip(rounds, per_round):
    mean = sum(vals) / len(vals)
    cf, mf = r.get("clip_fraction"), r.get("masked_fraction")
    ok = cf is not None and mf is not None and abs(cf - mean) <= 1e-6 and abs(mf - mean) <= 1e-6
    rows.append({"rollout_id": r["rollout_id"], "pg_clipfrac_steps": vals, "mean": mean,
                 "clip_fraction": cf, "masked_fraction": mf, "ok": ok})
checks["per_round_non_null_and_consistent"] = len(rows) == 3 and all(x["ok"] for x in rows)
ok = rc in ("0", "2") and all(v for k, v in checks.items() if k != "exit_code")
out = {"checks": checks, "rounds": rows, "pass": ok}
open(f"{d}/check.json", "w").write(json.dumps(out, indent=1)); print(json.dumps(out))
