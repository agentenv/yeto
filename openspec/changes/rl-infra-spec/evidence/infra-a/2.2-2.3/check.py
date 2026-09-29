"""Evaluate the pre-registered 2.2 / X9 conditions (plan.md). usage: check.py <A tape> <B tape> <C tape>"""
import json, sys

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def start(t):
    return next((e for e in t if e.get("event") == "rl_driver_start"), {})

def trained(t):
    return [e for e in t if e.get("event") == "rl_round_trained"]

def rounds(t):
    return [e for e in t if e.get("event") == "rl_local_round"]

def phases(t):
    return [e["phase"] for e in t if e.get("event") == "rl_driver_phase" and e["phase"] in ("generate", "train", "sync", "publish")]

def pub_ok(t):
    applies = {e["policy_version"]: e["sync/global_policy_hash"] for e in t if e.get("event") == "rl_policy_apply"}
    for e in t:
        if e.get("event") == "rl_publication":
            if not e.get("sync/publication_members"):
                return False
            v = e["policy_version"]
            if v in applies and not e["rl/policy_token"].endswith(applies[v]):
                return False
    return True

A, B, C = (load(p) for p in sys.argv[1:4])
res = {}
fail = lambda t: any(e.get("event") == "rl_strict_failure" for e in t)
res["c1_B_partitioned_no_failure"] = start(B).get("execution_mode") == "partitioned-serial" and not fail(B)
tA, tB = trained(A), trained(B)
res["rounds_A_B"] = [len(tA), len(tB)]
key = lambda e: (e["rollout_id"], e["trained_sample_ids_sha256"], e["trained_groups"], e["trained_samples"])
res["c2_sample_ids_equal"] = len(tA) == len(tB) > 0 and [key(e) for e in tA] == [key(e) for e in tB]
lr = lambda t: [len(r.get("applied_lrs") or []) for r in rounds(t)]
res["applied_lrs_len_A_B"] = [lr(A), lr(B)]
res["c3_optimizer_timing"] = phases(A) == phases(B) and lr(A) == lr(B) and phases(A).count("train") == len(tA)
res["c4_publication"] = pub_ok(A) and pub_ok(B)
res["c5_injected"] = sum(e.get("event") == "rl_fault_injected" for e in C) == sum(e.get("event") == "rl_publication" for e in C) > 0
# c6: generation after publication; no generate between publish start and rl_publication
ok6, pub_t, open_pub = True, {}, False
for e in C:
    if e.get("event") == "rl_driver_phase" and e.get("phase") == "publish":
        open_pub = True
    elif e.get("event") == "rl_publication":
        open_pub = False
        pub_t[e["policy_version"]] = e["time_unix"]
    elif e.get("event") == "rl_driver_phase" and e.get("phase") == "generate":
        r = e["rollout_id"]
        if open_pub or r not in pub_t or e["time_unix"] <= pub_t[r]:
            ok6 = False
res["c6_no_generation_before_publication"] = ok6 and bool(pub_t)
res["c7_C_no_failure"] = not fail(C) and start(C).get("execution_mode") == "partitioned-serial"
res["baseline_A"] = [{k: r.get(k) for k in ("local_round_id", "reward_mean", "grad_norm", "rollout_seconds", "train_seconds")} for r in rounds(A)]
res["baseline_B"] = [{k: r.get(k) for k in ("local_round_id", "reward_mean", "grad_norm", "rollout_seconds", "train_seconds")} for r in rounds(B)]
res["PASS_2_2"] = all(res[k] for k in ("c1_B_partitioned_no_failure", "c2_sample_ids_equal", "c3_optimizer_timing", "c4_publication"))
res["PASS_X9"] = all(res[k] for k in ("c5_injected", "c6_no_generation_before_publication", "c7_C_no_failure"))
print(json.dumps(res, indent=1, sort_keys=True))
