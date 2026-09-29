"""Pre-declared G1 criteria (plan.md): check_g1.py <run-name> <rundir> [rounds]. Prints JSON, exit 0 iff pass."""
import ast, json, math, re, sys
from pathlib import Path
name, d = sys.argv[1], Path(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
KEYS = {
    "observe": ["tis", "tis_abs", "mismatch_outside_0p5_5", "tis_abs_p50", "tis_abs_p90", "tis_abs_p99",
                "ois", "train_rollout_kl", "ess_ratio"],
    "tis": ["tis", "tis_abs", "tis_clipfrac", "ois", "train_rollout_kl"],
    "icepop": ["tis", "tis_abs", "tis_clipfrac", "ois", "train_rollout_kl"],
    "opsm-trainer": ["opsm_clipfrac", "train_rollout_kl"],
    "mis": ["ois"], "mis-mask": ["ois", "mis_tis_mask_fraction_low", "mis_tis_mask_fraction_high"],
}
res = {"run": name, "checks": {}}
c = res["checks"]
rc = (d / "rc").read_text().strip() if (d / "rc").exists() else "missing"
c["rc0"] = rc == "0"
log = (d / "miles.log").read_text(errors="replace") if (d / "miles.log").exists() else ""
steps = {}
for m in re.finditer(r"log_utils\.py:\d+ - step (\d+): (\{.*\})", log):
    steps[int(m.group(1))] = ast.literal_eval(m.group(2))
res["steps"] = steps
c["train_steps"] = sorted(steps) == list(range(rounds))
want = KEYS[name]
missing = {s: [k for k in want if f"train/{k}" not in v] for s, v in steps.items()}
if name.startswith("mis"):
    for s, v in steps.items():
        if not any(k.startswith("train/mis_") for k in v):
            missing[s].append("mis_*")
c["keys_present"] = bool(steps) and not any(missing.values())
res["missing"] = missing
c["keys_finite"] = bool(steps) and all(
    isinstance(v.get(f"train/{k}"), (int, float)) and math.isfinite(v[f"train/{k}"])
    for v in steps.values() for k in want if f"train/{k}" in v)
c["no_invariant_errors_in_log"] = not re.search(r"StrictRlInvariantError|PolicyIdentityError|AlgorithmMismatchError", log)
ev_path = d / "island-0" / "events.jsonl"
ev = [json.loads(x) for x in ev_path.read_text().splitlines()] if ev_path.exists() else []
kinds = [e.get("event") for e in ev]
rounds_ev = [e for e in ev if e.get("event") == "rl_local_round"]
pubs = [e for e in ev if e.get("event") == "rl_publication"]
c["rounds_events"] = len(rounds_ev) == rounds
c["publications_with_tokens"] = len(pubs) == rounds + 1 and all(p.get("rl/policy_token") for p in pubs)
c["no_failure_events"] = not any(k in ("rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch",
                                        "rl_algorithm_island_rejected") for k in kinds)
c["grad_norm_finite"] = bool(rounds_ev) and all(isinstance(e.get("grad_norm"), (int, float)) and
                                                math.isfinite(e["grad_norm"]) for e in rounds_ev)
exp = json.loads((d / "expected.json").read_text())
sel = [e for e in ev if e.get("event") == "rl_engine_selected"]
c["selected_sha_and_allowances"] = bool(sel) and sel[0].get("rl/algorithm_spec_sha256") == exp["sha256"] \
    and sorted(sel[0].get("rl/unverified_mechanisms", [])) == exp["allow"]
if len(sys.argv) > 4 and sys.argv[4] == "no-sync":
    c["outer_sync_false_and_flagged"] = bool(sel) and sel[0].get("rl/outer_sync") is False \
        and sel[0].get("rl/contains_unverified_mechanisms") is True
res["event_counts"] = {k: kinds.count(k) for k in sorted(set(kinds), key=str)}
res["pass"] = all(c.values())
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if res["pass"] else 1)
