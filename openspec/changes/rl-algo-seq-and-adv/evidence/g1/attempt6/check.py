"""Evaluate pre-declared G1 criteria for one attempt-6 run. usage: check.py <run>"""
import glob, json, math, re, sys
run = sys.argv[1]
log = open(f"{run}/launch.log", errors="replace").read()
ev = []
for f in glob.glob(f"{run}/events/*.jsonl"):
    ev += [json.loads(l) for l in open(f) if l.strip()]
names = [e.get("event") for e in ev]
rounds = [e for e in ev if e.get("event") == "rl_round_trained"]
sel = next((e for e in ev if e.get("event") == "rl_engine_selected"), {})
errs = re.findall(r"StrictRlInvariantError|RoundFailedError|AdvantageTransformError|RewardPipelineError|CapabilityMismatch|AlgorithmSpecError", log)
steps = re.findall(r"'train/pg_clipfrac': ([^,]+),.*?'train/grad_norm': ([^,}]+)", log)
steps = list(dict.fromkeys(steps))  # log lines appear twice (log_utils + model)
gn = [float(g) for _, g in steps]
print(json.dumps({
    "run": run,
    "finalized": "rl_learner_finalized" in names or "learner 0 finalized" in log,
    "rl_round_trained": len(rounds),
    "errors": sorted(set(errs)),
    "steps(clipfrac,grad_norm)": steps,
    "grad_norm_finite": all(math.isfinite(x) for x in gn) and bool(gn),
    "unverified": sel.get("rl/unverified_mechanisms"), "outer_sync": sel.get("rl/outer_sync"),
    "round_fields": [{k: r.get(k) for k in ("rollout_id", "masked_fraction", "clip_fraction", "nonzero_advantages")} for r in rounds],
    "zero_grad_events": [e for e in ev if "zero_grad" in str(e.get("event"))],
    "argv_gspo": "advantage_estimator ............................ gspo" in log or bool(re.search(r"advantage_estimator \.+ gspo", log)),
    "argv_kl_coef": re.findall(r"kl_coef \.+ ([0-9.e-]+)", log)[:1],
    "argv_normalize_advantages": re.findall(r"normalize_advantages \.+ (\w+)", log)[:1],
    "transform_events": re.findall(r'"nonzero_advantages": (\d+), "rollout_id": (\w+)[^}]*"transform": "(\w+)"', log),
}, indent=1))
