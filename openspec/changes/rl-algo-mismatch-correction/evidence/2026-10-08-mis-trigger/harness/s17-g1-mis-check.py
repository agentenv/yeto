"""s17-g1-mis-check.py <rundir>: MIS-TRUNCATE-TRIGGER-PREREG-S17 criteria (validity as g1c 1-6 + effect)."""
import ast, json, math, re, sys
from pathlib import Path
d = Path(sys.argv[1]); allow = ["corrections:mis"]
sys.path.insert(0, str(d / "yeto"))
from yeto.rl.engine.algorithm import AlgorithmSpec
spec = AlgorithmSpec.from_json_file(str(d / "spec.json"))
c = {}
rc = (d / "rc").read_text().strip(); log = (d / "launch.log").read_text(errors="replace")
c["rc"] = rc == "0" or (rc == "2" and "is not fetchable over ssh" in log)
steps = {}
for m in re.finditer(r"log_utils\.py:\d+ - step (\d+): (\{.*\})", log):
    steps.setdefault(int(m.group(1)), ast.literal_eval(m.group(2)))
c["train_steps_0_2"] = sorted(steps) == [0, 1, 2]
evf = sorted((d / "runs").glob("*/events/*.jsonl"))
ev = [json.loads(x) for x in evf[0].read_text().splitlines() if x.strip()] if evf else []
rounds = [e for e in ev if e.get("event") == "rl_local_round"]
pubs = [e for e in ev if e.get("event") == "rl_publication"]
sel = [e for e in ev if e.get("event") == "rl_engine_selected"]
c["rounds_1_3_traj32_gradfinite"] = [e.get("local_round_id") for e in rounds] == [1, 2, 3] and \
    all(e.get("completed_trajectories") == 32 and math.isfinite(e.get("grad_norm", math.nan)) for e in rounds)
c["publications_v0_3_tokens"] = [e.get("policy_version") for e in pubs] == [0, 1, 2, 3] and all(p.get("rl/policy_token") for p in pubs)
c["no_failure_events"] = not any(e.get("event") in ("rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch",
                                                    "rl_algorithm_island_rejected") for e in ev)
c["no_invariant_error_in_log"] = "zero_grad_norm_with_nonzero_advantages" not in log
c["selected_sha_allow"] = bool(sel) and sel[0].get("rl/algorithm_spec_sha256") == spec.sha256() and \
    sorted(sel[0].get("rl/unverified_mechanisms", [])) == allow and sel[0].get("rl/outer_sync") is False
per = {k: {f: v.get(f"train/{f}") for f in ("mis_tis_truncate_fraction", "mis_tis_weight_before_bound",
       "mis_tis_weight_after_bound", "mis_is_ratio_max_final", "grad_norm", "train_rollout_logprob_abs_diff")} for k, v in steps.items()}
c["EFFECT"] = any((p["mis_tis_truncate_fraction"] or 0) > 0 and p["mis_tis_weight_after_bound"] <= p["mis_tis_weight_before_bound"]
                  for p in per.values())
res = {"checks": c, "pass": all(c.values()), "sha": spec.sha256(), "per_step": per, "rc": rc,
       "rounds": [{k: e.get(k) for k in ("local_round_id", "reward_mean", "grad_norm", "delta_l2_norm", "completed_trajectories", "time_unix")} for e in rounds],
       "event_counts": {k: [e.get("event") for e in ev].count(k) for k in sorted({str(e.get("event")) for e in ev})}}
(d / "check.json").write_text(json.dumps(res, indent=1, default=str)); print("PASS" if res["pass"] else "FAIL", [k for k, v in c.items() if not v]); print(json.dumps(per, indent=0))
