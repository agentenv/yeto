"""S18 ARU-3 5.5: compare MB / M0 / M1 (pre-registered criteria: infra-drafts/S18-ARU3-PRELAUNCH-REVIEW.md §5).

Reads the island tape of each arm (raw data already on disk), writes compare.json next to this file.
"""
import collections
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

B = Path("/home/michael/work/s1-runs")
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "20261009a"
ARMS = {"MB": "s18-aru3-mb-20261009b", "M0": f"s18-aru3-m0-{SUFFIX}", "M1": f"s18-aru3-m1-{SUFFIX}"}  # MB rerun (a crashed at start)


def load(run):
    tapes = sorted(glob.glob(str(B / run / "tape-direct/**/rl-island-0.jsonl"), recursive=True))
    src = tapes[0] if tapes else str(B / run / "launch.log")
    ev = []
    for line in open(src, errors="replace"):
        i = line.find("YETO_RL_EVENT ")
        i = i + 14 if i >= 0 else line.find("{")
        if i < 0:
            continue
        try:
            e = json.loads(line[i:])
        except ValueError:
            continue
        if isinstance(e, dict) and "event" in e:
            ev.append(e)
    return src, ev


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def summ(run):
    src, ev = load(run)
    by = collections.defaultdict(list)
    for e in ev:
        by[e["event"]].append(e)
    loc = sorted(by["rl_local_round"], key=lambda e: e.get("local_round_id", e.get("rollout_id", 0)))
    t = [e.get("time_unix") for e in loc]
    round_s = [b - a for a, b in zip(t, t[1:]) if a is not None and b is not None]
    spans = collections.defaultdict(dict)
    for s in by["rl_timeline_span"]:
        if s.get("end") is not None and s.get("start") is not None:
            spans[s.get("task")][s.get("rollout_id")] = s["end"] - s["start"]
    gen = spans.get("generate", {})
    train = spans.get("train", {})
    traj = by["rl_trajectory_reward"]
    toks = [sum(x.get("turn_completion_tokens") or []) for x in traj]
    rew = [float(x.get("reward") or 0) for x in traj]
    turns = [x.get("turns") for x in traj]
    walls = [x["trajectory_ended_at"] - x["trajectory_started_at"] for x in traj
             if x.get("trajectory_ended_at") and x.get("trajectory_started_at")]
    loads = by["rl_load_sample"]
    kv = [l["kv_used_tokens"] / l["kv_capacity_tokens"] for l in loads
          if l.get("kv_used_tokens") is not None and l.get("kv_capacity_tokens")]
    first_round = min((e.get("time_unix") for e in by["rl_heartbeat"] if e.get("phase") not in (None, "startup")), default=None)
    gpu = [g.get("util_pct") for l in by["rl_resource_sample"] if l.get("phase") != "startup"
           for g in (l.get("gpus") or []) if g.get("util_pct") is not None]
    gpu_mem_peak = max((g.get("mem_used_mb") or 0 for l in by["rl_resource_sample"] for g in (l.get("gpus") or [])), default=None)
    cut = {c["rollout_id"]: c for c in by["rl_rollout_cutoff"]}
    car = {c["rollout_id"]: c for c in by["rl_rollout_carry_over"]}
    trained = {r["rollout_id"]: r for r in by["rl_round_trained"]}
    gn = [loc_e.get("grad_norm") for loc_e in loc]
    infra_reasons = collections.Counter(str(x.get("end_reason"))[:80] for x in traj)
    carried = [x for x in traj if isinstance(x.get("started_rollout_id"), int)
               and x.get("started_rollout_id") < x.get("rollout_id", -1)]
    carry_keys = ("suspended_groups", "suspended_done_groups", "resumed_groups", "resumed_done_groups",
                  "over_age_cancelled_groups", "over_age_cancelled_samples", "over_age_cancelled_tokens",
                  "over_age_unknown_groups", "suspend_expired_trajectories", "policy_age_exceeded_trajectories",
                  "suspended_trajectories", "suspended_env_seconds", "suspend_retried_turns",
                  "cross_version_tokens", "trained_response_tokens", "cross_version_truncated_fraction",
                  "cross_version_scored_tokens", "cross_version_unscored_samples",
                  "cross_version_ratio_p50", "cross_version_ratio_p90", "cross_version_ratio_p99",
                  "cross_version_ratio_min", "cross_version_ratio_max", "trained_groups_with_older_tokens",
                  "engines_idle_confirmed", "carried_out_groups")
    per_round_carry = {r: {k: c.get(k) for k in carry_keys if k in c} for r, c in sorted(car.items())}
    tot = lambda k: sum(int(c.get(k) or 0) for c in car.values())
    return {
        "src": src, "rc": (B / run / "rc.txt").read_text().strip() if (B / run / "rc.txt").exists() else None,
        "rounds_trained": len(trained), "trajectories_trained": len(traj),
        "round_total_s": [round(x, 1) for x in round_s],
        "round_total_median_s": med(round_s), "round_total_median_s_ge1": med(round_s[1:]),
        "round_total_median_s_drop_slowest": med(sorted(round_s)[:-1]) if len(round_s) > 1 else None,
        "generate_s_by_round": {k: round(v, 1) for k, v in sorted(gen.items())},
        "train_s_by_round": {k: round(v, 1) for k, v in sorted(train.items())},
        "generate_median_s": med(gen.values()), "train_median_s": med(train.values()),
        "reward_mean": st.mean(rew) if rew else None, "reward_std": st.pstdev(rew) if rew else None,
        "reward_se": (st.pstdev(rew) / math.sqrt(len(rew))) if rew else None, "nonzero_rewards": sum(r > 0 for r in rew),
        "tokens_mean": st.mean(toks) if toks else None, "tokens_std": st.pstdev(toks) if toks else None,
        "tokens_se": (st.pstdev(toks) / math.sqrt(len(toks))) if toks else None,
        "tokens_p10_p50_p90": [q(toks, p) for p in (.1, .5, .9)],
        "turns_mean": st.mean([x for x in turns if x is not None]) if any(x is not None for x in turns) else None,
        "traj_wall_s_p50_p90_max": [q(walls, .5), q(walls, .9), q(walls, 1.0)],
        "kv_peak_ratio": max(kv) if kv else None,
        "queued_requests_max": max((l.get("queued_requests") or 0) for l in loads) if loads else None,
        "gpu_util_mean": st.mean(gpu) if gpu else None, "gpu_mem_peak_mb": gpu_mem_peak,
        "end_reasons": infra_reasons.most_common(12),
        "cutoff_discarded_trajectories": sum(int(c.get("discarded_trajectories") or 0) for c in cut.values()),
        "cutoff_discarded_tokens": sum(int(c.get("discarded_tokens") or 0) for c in cut.values()),
        "cutoff_discarded_tokens_unknown_rounds": sum(1 for c in cut.values() if c.get("discarded_tokens") is None),
        "cutoff_events": {r: {k: v for k, v in c.items() if k in ("submitted_groups", "target_groups", "discarded_groups",
                                                                  "discarded_trajectories", "discarded_tokens",
                                                                  "filtered_groups", "discarded_unknown_groups")}
                          for r, c in sorted(cut.items())},
        "carry_per_round": per_round_carry,
        "carry_totals": {k: tot(k) for k in ("suspended_groups", "resumed_groups", "over_age_cancelled_groups",
                                             "over_age_cancelled_samples", "over_age_cancelled_tokens",
                                             "over_age_unknown_groups", "suspend_expired_trajectories",
                                             "policy_age_exceeded_trajectories", "suspended_trajectories",
                                             "suspend_retried_turns", "cross_version_tokens",
                                             "trained_response_tokens", "trained_groups_with_older_tokens")},
        "suspended_env_seconds_total": sum(float(c.get("suspended_env_seconds") or 0) for c in car.values()),
        "carried_trained_trajectories": len(carried),
        "carried_trajectory_examples": [{k: x.get(k) for k in ("rollout_id", "started_rollout_id", "policy_versions",
                                                               "task_id", "reward", "suspended_seconds")}
                                        for x in carried[:5]],
        "grad_norm": gn, "grad_norm_finite": all(g is not None and math.isfinite(g) for g in gn),
        "invariant_failed": len(by["rl_invariant_failed"]),
        "policy_age_warnings": len(by["rl_policy_age_warning"]), "policy_age_fallbacks": len(by["rl_policy_age_fallback"]),
        "trained_tis_clipfrac": [(trained[r].get("train_metrics") or {}).get("tis_clipfrac") for r in sorted(trained)],
    }


out = {arm: summ(run) for arm, run in ARMS.items()}
MB, M0, M1 = out["MB"], out["M0"], out["M1"]
crit = {}
crit["validity_rc0_6rounds"] = all(a["rc"] == "rc=0" and a["rounds_trained"] == 6 for a in out.values())
resumed_rounds = sum(1 for c in M1["carry_per_round"].values() if (c.get("resumed_groups") or 0) > 0)
crit["validity_m1_resumed_rounds_ge3"] = resumed_rounds >= 3
crit["validity_m1_trained_groups_with_older_tokens"] = M1["carry_totals"]["trained_groups_with_older_tokens"] > 0
if MB["round_total_median_s"] and M1["round_total_median_s"]:
    crit["1_m1_round_median_le_0.8_mb"] = M1["round_total_median_s"] <= 0.8 * MB["round_total_median_s"]
    crit["1_ratio"] = M1["round_total_median_s"] / MB["round_total_median_s"]
m1_disc_traj = (M1["carry_totals"]["over_age_cancelled_groups"] * 4 + M1["carry_totals"]["suspend_expired_trajectories"]
                + M1["carry_totals"]["policy_age_exceeded_trajectories"])
crit["2_m1_discarded_trajectories"] = m1_disc_traj
crit["2_m0_discarded_trajectories"] = M0["cutoff_discarded_trajectories"]
crit["2_traj_ratio_le_0.5"] = (m1_disc_traj <= 0.5 * M0["cutoff_discarded_trajectories"]) if M0["cutoff_discarded_trajectories"] else None
crit["2_m1_known_discarded_tokens"] = M1["carry_totals"]["over_age_cancelled_tokens"] + M1["cutoff_discarded_tokens"]
crit["2_m1_unknown_token_groups"] = M1["carry_totals"]["over_age_unknown_groups"]
crit["2_m0_discarded_tokens"] = M0["cutoff_discarded_tokens"]
for key in ("reward", "tokens"):
    if MB[f"{key}_std"] is not None and M1[f"{key}_mean"] is not None:
        crit[f"3_{key}_m1_within_1sd_mb"] = abs(M1[f"{key}_mean"] - MB[f"{key}_mean"]) <= MB[f"{key}_std"]
        crit[f"3_{key}_diff_over_sd"] = (M1[f"{key}_mean"] - MB[f"{key}_mean"]) / MB[f"{key}_std"] if MB[f"{key}_std"] else None
fr = [c.get("cross_version_truncated_fraction") for c in M1["carry_per_round"].values()]
crit["5_no_round_ge_0.5"] = all(f is None or f < 0.5 for f in fr)
crit["5_fractions"] = fr
crit["6_numeric"] = all(a["grad_norm_finite"] and a["invariant_failed"] == 0 for a in out.values())
result = {"arms": out, "criteria": crit}
(Path(__file__).parent / "compare.json").write_text(json.dumps(result, indent=1, default=str))
print(json.dumps(crit, indent=1, default=str))
