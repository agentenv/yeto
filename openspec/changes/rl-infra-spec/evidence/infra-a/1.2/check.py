"""Evaluate the pre-registered 1.2 success conditions (plan.md) on pulled island tapes."""
import json, sys
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
from yeto.rl.engine.algorithm import AlgorithmSpec
d = sys.argv[1]
tapes = {i: [json.loads(l) for l in open(f"{d}/rl-island-{i}.jsonl") if l.strip()] for i in (0, 1)}
res = {}
res["no_strict_failure"] = all(not any(e.get("event") == "rl_strict_failure" for e in t) for t in tapes.values())
rounds = {i: [e for e in t if e.get("event") == "rl_local_round"] for i, t in tapes.items()}
res["rounds_per_island"] = {i: len(r) for i, r in rounds.items()}
res["at_least_one_round"] = all(len(r) >= 1 for r in rounds.values())
starts = {i: next((e for e in t if e.get("event") == "rl_driver_start"), {}) for i, t in tapes.items()}
res["execution_mode"] = {i: s.get("execution_mode") for i, s in starts.items()}
res["algo_hash_ok"] = all(s.get("rl/algorithm_spec_sha256") == AlgorithmSpec().sha256() for s in starts.values())
applies = {i: {e["policy_version"]: e["sync/global_policy_hash"] for e in t if e.get("event") == "rl_policy_apply"} for i, t in tapes.items()}
common = sorted(set(applies[0]) & set(applies[1]))
res["apply_versions_common"] = common
res["apply_hash_equal_across_islands"] = bool(common) and all(applies[0][v] == applies[1][v] for v in common)
pub_ok = True
for i, t in tapes.items():
    for e in t:
        if e.get("event") == "rl_publication":
            v = e["policy_version"]; tok = e["rl/policy_token"]
            if v in applies[i] and not tok.endswith(applies[i][v]):
                pub_ok = False
            if not e.get("sync/publication_members"):
                pub_ok = False
res["publication_matches_apply_and_members"] = pub_ok
phases = {i: [e.get("phase") for e in t if e.get("event") == "rl_driver_phase"] for i, t in tapes.items()}
res["phase_sequence_ok"] = all(
    any(p[k:k + 4] == ["generate", "onload", "train", "sync"] or p[k:k + 3] == ["generate", "train", "sync"] for k in range(len(p))) and "publish" in p
    for p in phases.values())
res["baseline"] = {i: [{k: r.get(k) for k in ("local_round_id", "reward_mean", "reward_std", "loss", "grad_norm", "rollout_seconds", "train_seconds")} for r in rounds[i]] for i in rounds}
res["PASS_conditions_1_to_4"] = all([res["no_strict_failure"], res["at_least_one_round"], res["apply_hash_equal_across_islands"], res["publication_matches_apply_and_members"], res["algo_hash_ok"], res["phase_sequence_ok"], all(v == "colocated-serial" for v in res["execution_mode"].values())])
print(json.dumps(res, indent=1, sort_keys=True))
