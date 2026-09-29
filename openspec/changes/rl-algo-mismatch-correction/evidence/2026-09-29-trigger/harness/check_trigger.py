"""check_trigger.py <run> <rundir> <steps_per_round> <allow...>: plan.md criteria (validity + effect)."""
import ast, json, math, re, sys
from pathlib import Path
run, d, spr, allow = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sorted(sys.argv[4:])
sys.path.insert(0, "/home/michael/work/algo-1a")
from yeto.rl.engine.algorithm import AlgorithmSpec
spec = AlgorithmSpec.from_json_file(str(d / "spec.json"))
c = {}
rc = (d / "rc").read_text().strip(); log = (d / "launch.log").read_text(errors="replace")
c["rc"] = rc == "0" or (rc == "2" and "is not fetchable over ssh" in log)
steps = {int(m.group(1)): ast.literal_eval(m.group(2)) for m in re.finditer(r"log_utils\.py:\d+ - step (\d+): (\{.*\})", log)}
c["train_steps"] = sorted(steps) == list(range(3 * spr))
tape = d / "tape.jsonl"
ev = [json.loads(x) for x in tape.read_text().splitlines()] if tape.exists() else []
rounds = [e for e in ev if e.get("event") == "rl_local_round"]
pubs = [e for e in ev if e.get("event") == "rl_publication"]
sel = [e for e in ev if e.get("event") == "rl_engine_selected"]
c["rounds_and_receipt_stats"] = [e.get("local_round_id") for e in rounds] == [1, 2, 3] and \
    all(e.get("completed_trajectories") == 32 and math.isfinite(e.get("grad_norm", math.nan)) for e in rounds) and \
    [e.get("policy_version") for e in pubs] == [0, 1, 2, 3] and all(p.get("rl/policy_token") for p in pubs)
c["no_failure_events"] = not any(e.get("event") in ("rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch",
                                                    "rl_algorithm_island_rejected") for e in ev)
c["selected"] = bool(sel) and sel[0].get("rl/algorithm_spec_sha256") == spec.sha256() and \
    sorted(sel[0].get("rl/unverified_mechanisms", [])) == allow and sel[0].get("rl/outer_sync") is False
def vals(k): return [v.get(f"train/{k}") for v in steps.values()]
eff = {"tis": ["tis_clipfrac"], "icepop": ["tis_clipfrac"], "opsm-trainer": ["opsm_clipfrac"],
       "mis-mask": ["mis_tis_mask_fraction_low", "mis_tis_mask_fraction_high"]}[run]
per_step = [sum(float(v.get(f"train/{k}", 0) or 0) for k in eff) for v in steps.values()]
c["metrics_finite"] = bool(steps) and all(all(isinstance(v.get(f"train/{k}"), (int, float)) and math.isfinite(v[f"train/{k}"]) for k in eff) for v in steps.values())
c["EFFECT_fraction_gt_0"] = any(x > 0 for x in per_step)
res = {"checks": c, "pass": all(c.values()), "effect_per_step": per_step,
       "event_counts": {k: [e.get("event") for e in ev].count(k) for k in sorted({e.get("event") for e in ev}, key=str)}}
(d / "check.json").write_text(json.dumps(res, indent=1, default=str)); print(run, "PASS" if res["pass"] else "FAIL", [k for k, v in c.items() if not v], per_step)
