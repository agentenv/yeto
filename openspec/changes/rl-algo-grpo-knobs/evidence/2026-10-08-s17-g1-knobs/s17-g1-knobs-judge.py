#!/usr/bin/env python3
"""S17 G1 (N3) judge for rl-algo-grpo-knobs 8.4 (two islands strict-avg, combined clip-higher+token+overlong).
usage: judge.py <run dir> <expected sha>. Criteria fixed in infra-drafts/S17-G1-PRELAUNCH-REVIEW.md §6 D."""
import json, math, sys
from pathlib import Path
R, SHA = Path(sys.argv[1]), sys.argv[2]
log = (R / "launch.log").read_text(errors="replace")
ev, seen = [], set()
for line in log.splitlines():
    at = line.find("YETO_RL_EVENT ")
    if at < 0: continue
    s = line[at + 14:].strip()
    if s in seen: continue
    seen.add(s)
    try: ev.append(json.loads(s))
    except ValueError: pass
E = lambda n: [e for e in ev if e.get("event") == n]
c, info = {}, {}
c["rc0"] = (R / "rc.txt").is_file() and (R / "rc.txt").read_text().strip() == "rc=0"
sel = E("rl_engine_selected")
info["selected"] = [(e.get("island_id"), e.get("rl/algorithm_spec_sha256"), e.get("rl/unverified_mechanisms"), e.get("rl/outer_sync")) for e in sel]
c["both_islands_same_sha"] = sorted(e.get("island_id") for e in sel) == [0, 1] and all(e.get("rl/algorithm_spec_sha256") == SHA for e in sel) \
    and all(not e.get("rl/unverified_mechanisms") for e in sel)
rounds = E("rl_local_round")
c["three_rounds_each"] = all(sorted(e["local_round_id"] for e in rounds if e.get("island_id") == i) == [1, 2, 3] for i in (0, 1))
hashes = {}
for e in E("rl_policy_apply"):
    hashes.setdefault(e["policy_version"], {}).setdefault(e["island_id"], set()).add(e.get("sync/global_policy_hash"))
info["apply_hashes"] = {v: {i: sorted(h) for i, h in d.items()} for v, d in sorted(hashes.items())}
c["post_sync_hash_equal"] = all(len(d) == 2 and len(set().union(*d.values())) == 1 for v, d in hashes.items() if v >= 1) and \
    sorted(v for v in hashes if v >= 1) == [1, 2, 3]
c["no_failure_events"] = not any(e.get("event") in ("rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch",
                                                    "rl_algorithm_island_rejected") for e in ev) and \
    "zero_grad_norm_with_nonzero_advantages" not in log
fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
c["numeric_finite"] = bool(rounds) and all(fin(e.get("grad_norm")) and fin(e.get("delta_l2_norm")) for e in rounds)
trained = E("rl_round_trained")
info["effective_samples"] = [{k: e.get(k) for k in ("island_id", "local_round_id", "rollout_id", "completed_trajectories", "trained_samples",
                              "filtered_samples", "shaped_samples", "nonzero_advantages") if k in e} for e in trained]
info["rounds"] = [{k: e.get(k) for k in ("island_id", "local_round_id", "reward_mean", "grad_norm", "delta_l2_norm", "completed_trajectories", "clip_fraction")} for e in rounds]
c["effective_samples_recorded"] = len(trained) >= 6 and all(e.get("filtered_samples") is not None for e in trained)
res = {"checks": c, "pass": all(c.values()), "info": info}
print(json.dumps(res, indent=1, default=str))
