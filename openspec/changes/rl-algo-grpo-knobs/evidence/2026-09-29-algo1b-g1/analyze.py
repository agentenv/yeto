"""Apply plan.md criteria to out/<mech> (no thresholds beyond the plan)."""
import ast, json, math, re, sys
from pathlib import Path
OUT = Path(__file__).parent / "out"
KEYS = {"baseline": ["pg_loss", "grad_norm"], "clip_higher": ["pg_clipfrac", "pg_loss"],
        "dual_clip": ["pg_clipfrac", "pg_loss"], "token": ["pg_clipfrac", "pg_loss"],
        "drgrpo": ["pg_clipfrac", "pg_loss"], "kl_k3": ["kl_loss"], "entropy": ["entropy_loss"],
        "over_sampling": [], "overlong_penalty": []}
BAD = re.compile(r"zero_grad_norm_with_nonzero_advantages|nonfinite_grad_norm|StrictRlInvariantError|"
                 r"policy token mismatch|PolicyTokenMismatch|RoundFailedError|Traceback")
report = {}
for mech, keys in KEYS.items():
    d = OUT / mech
    r = {"present": d.exists()}
    if not d.exists():
        report[mech] = r; continue
    res = json.loads((d / "g1_result.json").read_text()) if (d / "g1_result.json").exists() else {}
    r.update(res)
    log = (d / "miles.log").read_text(errors="replace") if (d / "miles.log").exists() else ""
    ev = [json.loads(l) for l in (d / "island-0" / "events.jsonl").read_text().splitlines()] \
        if (d / "island-0" / "events.jsonl").exists() else []
    rounds = [e for e in ev if e["event"] == "rl_local_round"]
    r["rounds"] = len(rounds)
    r["round_train_seconds"] = [e.get("train_seconds") for e in rounds]
    r["round_rollout_seconds"] = [e.get("rollout_seconds") for e in rounds]
    r["bad_lines"] = sorted({m.group(0) for m in BAD.finditer(log)})
    steps = []
    for line in log.splitlines():
        if "model.py" in line and "step " in line and "{'train/" in line:
            steps.append(ast.literal_eval(line[line.index("{'train/"):].strip()))
    r["train_steps"] = len(steps)
    vals = {k: [s.get(f"train/{k}") for s in steps] for k in keys}
    r["metrics"] = vals
    r["metrics_ok"] = all(v and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                          for v in vals.values())
    extra = {}
    if mech == "over_sampling":
        extra = {k: [e.get(k) for e in rounds] for k in ("completed_groups", "dynamic_filter_dropped_groups",
                 "dynamic_filter_generated_groups", "dynamic_filter_replacement_attempts", "completed_trajectories")}
        r["metrics_ok"] = all(x is not None for x in extra["completed_groups"])
    if mech == "overlong_penalty":
        shaping = [ln for ln in log.splitlines() if "rl_reward_shaping" in ln]
        extra = {"rl_reward_shaping_lines": len(shaping), "sample": shaping[:2]}
        parsed = []
        for ln in shaping:
            try:
                parsed.append(json.loads(ln[ln.index("{"):]))
            except Exception:
                pass
        extra["parsed"] = parsed[:3]
        r["metrics_ok"] = bool(parsed) and all(
            p["raw_reward"]["mean"] is not None and p["shaped_reward"]["mean"] is not None for p in parsed)
    r["extra"] = extra
    r["pass"] = (res.get("rc") == 0 and r["rounds"] == 3 and not r["bad_lines"] and r["metrics_ok"])
    report[mech] = r
(OUT.parent / "g1_report.json").write_text(json.dumps(report, indent=1, default=str))
for m, r in report.items():
    print(m, "PASS" if r.get("pass") else "FAIL", {k: r.get(k) for k in ("rc", "rounds", "train_steps", "seconds", "peak_gpu_mib", "bad_lines")})
