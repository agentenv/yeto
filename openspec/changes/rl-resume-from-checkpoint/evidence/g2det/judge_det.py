"""S17 G2det (N15): per-round trainer numbers of the G2 / G2det runs, side by side.
usage: python judge_det.py <s1-runs dir> > g2det-judge.json"""
import glob, json, sys
B = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = {"g2-a": "s17-g2-a", "g2-c2": "s17-g2-c2", "g2-c3": "s17-g2-c3", "g2-b2": "s17-g2-b2",
        "det-a": "s17-g2det-a", "det-c": "s17-g2det-c", "det-e2": "s17-g2det-e2",
        "det-f1": "s17-g2det-f1", "det-f2": "s17-g2det-f2"}
out = {}
for name, run in RUNS.items():
    for tape in sorted(glob.glob(f"{B}/{run}/tape-direct/*/l0/rank0/rl-island-0*.jsonl")):
        seg = name + (".inc1" if ".inc1" in tape else "")
        rec = out.setdefault(seg, {"rounds": {}, "resume": None, "cuts": {}})
        for line in open(tape):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ev = e.get("event")
            if ev == "rl_round_trained":
                m = e.get("train_metrics") or {}
                rec["rounds"][e["rollout_id"]] = {
                    "ppo_kl": m.get("ppo_kl"), "train_rollout_kl": m.get("train_rollout_kl"),
                    "train_rollout_logprob_abs_diff": m.get("train_rollout_logprob_abs_diff"),
                    "resp_len_mean": e.get("resp_len_mean"), "ids": e.get("trained_sample_ids_sha256"),
                    "lrs": e.get("applied_lrs"), "step_seconds": e.get("step_seconds")}
            elif ev == "rl_resume":
                rec["resume"] = {k: e.get(k) for k in ("rollout_id", "cut_id", "policy_hash", "checks", "restore_s")}
            elif ev == "rl_cut_saved":
                rec["cuts"][e["rollout_id"]] = {k: e.get(k) for k in ("policy_hash", "final", "save_s", "store_commit_s")}
def same(a, b, rid):
    x, y = out.get(a, {}).get("rounds", {}).get(rid), out.get(b, {}).get("rounds", {}).get(rid)
    if x is None or y is None:
        return None
    return all(x[k] == y[k] for k in ("ppo_kl", "train_rollout_kl", "train_rollout_logprob_abs_diff", "resp_len_mean"))
out["_compare"] = {
    "det-a == g2-a (rounds 0-5)": [same("det-a", "g2-a", r) for r in range(6)],
    "det-c.inc1 == det-a (rounds 2-5)": [same("det-c.inc1", "det-a", r) for r in range(2, 6)],
    "det-c.inc1 == g2-c3.inc1 (rounds 2-5)": [same("det-c.inc1", "g2-c3.inc1", r) for r in range(2, 6)],
    "det-e2 == det-a (rounds 4-5)": [same("det-e2", "det-a", r) for r in (4, 5)],
    "det-e2 == g2-c2.inc1 (round 4)": [same("det-e2", "g2-c2.inc1", 4)],
    "det-f2 == det-e2 (rounds 4-5)": [same("det-f2", "det-e2", r) for r in (4, 5)],
    "det-f2 == det-a (rounds 4-5)": [same("det-f2", "det-a", r) for r in (4, 5)],
    "g2-b2 == g2-a (rounds 3-5)": [same("g2-b2", "g2-a", r) for r in (3, 4, 5)],
}
json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
