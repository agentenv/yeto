"""Apply plan.md criteria to out/<mech> (no thresholds beyond the plan)."""
import ast, json, math, re, sys
from pathlib import Path
OUT = Path(__file__).parent / "out"
KEYS = {"baseline": ["pg_loss", "grad_norm"], "clip_higher": ["pg_clipfrac", "pg_loss"],
        "dual_clip": ["pg_clipfrac", "pg_loss"], "token": ["pg_clipfrac", "pg_loss"],
        "drgrpo": ["pg_clipfrac", "pg_loss"], "kl_k3": ["kl_loss"], "entropy": ["entropy_loss"],
        "over_sampling": [], "overlong_penalty": [], "no_std": ["grad_norm"]}
BAD = re.compile(r"zero_grad_norm_with_nonzero_advantages|nonfinite_grad_norm|StrictRlInvariantError|"
                 r"policy token mismatch|PolicyTokenMismatch|RoundFailedError|Traceback")
# Benign chained exception (review decision, added after attempt 2; the
# original report is kept as g1_report.json): SGLang logs
# "post-warmup freeze_gc failed" when its own /freeze_gc call races server
# startup (ConnectionRefused -> MaxRetryError -> requests ConnectionError),
# catches it and keeps serving. A Traceback whose chain contains only that
# is not an invariant failure; any other Traceback still fails.
BENIGN_CONTEXT = "post-warmup freeze_gc failed"


def _malignant_tracebacks(log: str) -> int:
    lines = log.splitlines()
    bad = 0
    for i, ln in enumerate(lines):
        if "Traceback (most recent call last)" not in ln:
            continue
        window = "\n".join(lines[max(0, i - 80):i + 60])
        if BENIGN_CONTEXT in window and "/freeze_gc" in window:
            continue
        bad += 1
    return bad


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
    r["bad_lines"] = sorted({m.group(0) for m in BAD.finditer(log)} - {"Traceback"})
    if _malignant_tracebacks(log):
        r["bad_lines"].append("Traceback")
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
# g1c paired step-1 criterion (plan.md, fixed before the run): step-1 raw_reward
# equals baseline's (valid pair) and step-1 train/grad_norm differs from baseline's.
import re as _re
def _first(mech, key):
    p = OUT / mech / "miles.log"
    if not p.exists():
        return None
    m = _re.search(r"'" + _re.escape(key) + r"': ([-0-9.e]+)", p.read_text(errors="replace"))
    return float(m.group(1)) if m else None
if "baseline" in report:
    for mech in ("token", "no_std"):
        if mech in report and report[mech].get("present"):
            same_input = _first(mech, "rollout/raw_reward") == _first("baseline", "rollout/raw_reward")
            differs = _first(mech, "train/grad_norm") != _first("baseline", "train/grad_norm")
            report[mech]["paired_valid"] = same_input
            report[mech]["step1_grad_norm"] = _first(mech, "train/grad_norm")
            report[mech]["effective"] = bool(report[mech].get("pass") and same_input and differs)
(OUT.parent / (sys.argv[1] if len(sys.argv) > 1 else "g1_report_v2.json")).write_text(json.dumps(report, indent=1, default=str))
for m, r in report.items():
    print(m, "PASS" if r.get("pass") else "FAIL", {k: r.get(k) for k in ("rc", "rounds", "train_steps", "seconds", "peak_gpu_mib", "bad_lines")})
