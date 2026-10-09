"""S18 ARU-2 4.2: compare Miles arms MB / M0 / M1 from their island tapes (raw data on disk first)."""
import csv, glob, json, math, statistics as st, sys
from pathlib import Path

B = Path("/home/michael/work/s1-runs")
ARMS = {"MB": "s18-aru2-mb", "M0": "s18-aru2-m0", "M1": "s18-aru2-m1"}


def tape(run):
    c = sorted(glob.glob(str(B / run / "tape-direct/*/l0/rank0/rl-island-0.jsonl"))) or \
        sorted(glob.glob(str(B / run / "runs/*/modal-tape/*/l0/rank0/rl-island-0.jsonl"))) or \
        [str(B / run / "tape.jsonl")]
    return [json.loads(l) for l in open(c[0]) if l.strip()]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def nvml(run):
    p = B / run / "nvml.csv"
    vals = []
    if p.exists():
        for row in csv.reader(open(p)):
            try:
                vals.append(float(row[1].strip().split()[0]))
            except (IndexError, ValueError):
                pass
    return vals


out = {}
for arm, run in ARMS.items():
    ev = tape(run)
    loc = sorted([e for e in ev if e["event"] == "rl_local_round"], key=lambda e: e["local_round_id"])
    tr = {e["rollout_id"]: e for e in ev if e["event"] == "rl_round_trained"}
    cut = {e["rollout_id"]: e for e in ev if e["event"] == "rl_rollout_cutoff"}
    car = {e["rollout_id"]: e for e in ev if e["event"] == "rl_rollout_carry_over"}
    traj = [e for e in ev if e["event"] == "rl_trajectory_reward"]
    t = [e["time_unix"] for e in loc]
    round_s = [b - a for a, b in zip(t, t[1:])]
    gen = [e.get("rollout_seconds") for e in loc]
    train = [e.get("train_seconds") for e in loc]
    rew = [e.get("reward_mean") for e in loc]
    gn = [e.get("grad_norm") for e in loc]
    tis = [(tr[r].get("train_metrics") or {}).get("tis_clipfrac") for r in sorted(tr)]
    resp = [tr[r].get("resp_len_mean") for r in sorted(tr)]
    trunc = [tr[r].get("truncated_frac") for r in sorted(tr)]
    cvf = [car[r].get("cross_version_truncated_fraction") for r in sorted(car)]
    disc_cut = sum(int(c.get("discarded_tokens") or 0) for c in cut.values())
    disc_traj = sum(int(c.get("discarded_trajectories") or 0) for c in cut.values())
    disc_over = sum(int(c.get("over_age_discarded_tokens") or 0) for c in car.values())
    disc_unknown = sum(int(c.get("unknown_version_discarded_groups") or 0) for c in car.values())
    carried_tok = sum(int(c.get("carried_in_tokens") or 0) for c in car.values())
    carried_traj = sum(int(c.get("carried_in_trajectories") or 0) for c in car.values())
    cross = sum(int(c.get("cross_version_tokens") or 0) for c in car.values())
    trained_tok = sum(int(c.get("trained_response_tokens") or 0) for c in car.values())
    util = nvml(run)
    n = len(rew)
    out[arm] = {
        "rounds": n,
        "round_seconds": round_s, "round_seconds_median": med(round_s),
        "round_seconds_median_excl_first": med(round_s[1:]),
        "generation_seconds": gen, "generation_median": med(gen), "generation_median_excl_first": med(gen[1:]),
        "train_seconds_median": med(train),
        "reward_mean_per_round": rew,
        "reward_mean": st.fmean(rew) if rew else None,
        "reward_se": (st.stdev(rew) / math.sqrt(n)) if n > 1 else None,
        "grad_norm": gn, "grad_norm_all_positive_finite": all(g is not None and math.isfinite(g) and g > 0 for g in gn),
        "reward_all_finite": all(r is not None and math.isfinite(r) for r in rew),
        "invariant_failures": sum(1 for e in ev if e["event"] == "rl_invariant_failed"),
        "tis_clipfrac": tis, "resp_len_mean": resp, "truncated_frac": trunc,
        "cutoff_discarded_tokens": disc_cut, "cutoff_discarded_trajectories": disc_traj,
        "over_age_discarded_tokens": disc_over, "unknown_version_discarded_groups": disc_unknown,
        "carried_in_trajectories": carried_traj, "carried_in_tokens": carried_tok,
        "cross_version_tokens": cross, "trained_response_tokens_carry_rounds": trained_tok,
        "cross_version_token_share": (cross / trained_tok) if trained_tok else None,
        "cross_version_truncated_fraction": cvf,
        "carry_rounds": {r: {k: car[r].get(k) for k in ("carried_in_groups", "carried_out_groups",
                                                       "trained_groups_with_older_tokens",
                                                       "cross_version_unscored_samples")} for r in sorted(car)},
        "policy_age_events": [e["event"] for e in ev if e["event"] in ("rl_policy_age_warning", "rl_policy_age_fallback")],
        "gpu_util_mean": st.fmean(util) if util else None, "gpu_util_samples": len(util),
        "trajectory_records": len(traj),
        "started_earlier": sum(1 for e in traj if isinstance(e.get("started_rollout_id"), int)
                               and e["started_rollout_id"] < e["rollout_id"]),
    }

m0, m1 = out.get("M0"), out.get("M1")
if m0 and m1:
    j = {}
    j["time_ratio_median"] = m1["round_seconds_median"] / m0["round_seconds_median"]
    j["time_ratio_median_excl_first"] = m1["round_seconds_median_excl_first"] / m0["round_seconds_median_excl_first"]
    j["1_time_le_0.8"] = j["time_ratio_median"] <= 0.8
    m1_disc = m1["over_age_discarded_tokens"] + m1["cutoff_discarded_tokens"]
    j["m1_discarded_tokens"] = m1_disc
    j["m0_discarded_tokens"] = m0["cutoff_discarded_tokens"]
    j["2_discard_le_half"] = m1_disc <= 0.5 * m0["cutoff_discarded_tokens"]
    lo, hi = m0["reward_mean"] - 2 * m0["reward_se"], m0["reward_mean"] + 2 * m0["reward_se"]
    j["3_reward_within_2se"] = lo <= m1["reward_mean"] <= hi
    vals = [v for v in m1["cross_version_truncated_fraction"] if v is not None]
    j["4_no_fallback"] = all(v < 0.5 for v in vals) and "rl_policy_age_fallback" not in m1["policy_age_events"]
    j["5_health"] = all(out[a]["grad_norm_all_positive_finite"] and out[a]["reward_all_finite"]
                        and out[a]["invariant_failures"] == 0 for a in out)
    j["validity_carry_rounds"] = sum(1 for r in m1["carry_rounds"].values()
                                     if (r.get("carried_in_groups") or 0) > 0
                                     and (r.get("trained_groups_with_older_tokens") or 0) > 0)
    out["judgement"] = j
json.dump(out, open(B / "s18-aru2/compare.json", "w"), indent=1)
print(json.dumps({a: {k: v for k, v in d.items() if not isinstance(v, (list, dict))} for a, d in out.items() if a != "judgement"}, indent=1))
print(json.dumps(out.get("judgement"), indent=1))
