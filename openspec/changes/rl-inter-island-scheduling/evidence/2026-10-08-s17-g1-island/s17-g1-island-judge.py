#!/usr/bin/env python3
"""S17 G1 (N3) judge for rl-inter-island-scheduling 1.2 (relaxed per S17-OVERNIGHT-PLAN §一 7) + bandwidth.
usage: s17-g1-island-judge.py <island run dir> <baseline run dir>. Reads only saved raw files.
 B  bandwidth: kind:"transfer" rows on the head syncer tape -> per (direction,msg) bytes, seconds, GiB/s (sum and per row)
 R  reward: mean over local rounds of both islands m2 >= baseline mean m1 - 2*SE(baseline)
 H  numeric health: all rl_local_round grad_norm/delta_l2_norm/reward finite, grad_norm>0, delta>0;
    every policy_version applied by both islands has the same sync/global_policy_hash
 (exit 4/6 and catch-up zero weight: s15-island1b-judge.py / s15-island1b-pause-judge.py outputs, saved next to this)"""
import glob, json, math, re, statistics as st, sys
from pathlib import Path
R, BASE = Path(sys.argv[1]), Path(sys.argv[2])
out = {}
tape = [json.loads(l) for l in (R / "head/yeto-output/yeto-tape.jsonl").read_text().splitlines() if l.strip()] \
    if (R / "head/yeto-output/yeto-tape.jsonl").is_file() else []
tr = [r for r in tape if r.get("kind") == "transfer"]
groups = {}
for r in tr:
    g = groups.setdefault(f'{r["direction"]}:{r["msg"]}', {"n": 0, "bytes": 0, "seconds": 0.0, "per_row_gibps": []})
    g["n"] += 1; g["bytes"] += r["bytes"]
    if r.get("seconds") is not None:
        g["seconds"] += r["seconds"]
        if r["seconds"] > 0: g["per_row_gibps"].append(r["bytes"] / r["seconds"] / 2**30)
for g in groups.values():
    g["agg_MiBps"] = g["bytes"] / g["seconds"] / 2**20 if g["seconds"] > 0 else None
    p = sorted(g.pop("per_row_gibps")); g["per_row_MiBps_min_med_max"] = [x * 1024 for x in (p[0], p[len(p)//2], p[-1])] if p else None
    g["bytes_per_frame"] = g["bytes"] / g["n"]
out["B_transfer_rows"] = len(tr); out["B_groups"] = groups
out["B_pass"] = bool(tr) and all(isinstance(r.get("bytes"), int) and r["bytes"] > 0 and r.get("seconds") is not None and math.isfinite(r["seconds"]) for r in tr)
# island events from the launch log (dedup identical lines)
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
rounds = [e for e in ev if e.get("event") == "rl_local_round"]
x2 = [e["reward_mean"] for e in rounds if e.get("reward_mean") is not None]
bf = glob.glob(str(BASE / "runs/*/events/*.jsonl"))[0]
x1 = [json.loads(l)["reward_mean"] for l in open(bf) if '"rl_local_round"' in l]
m1, se1 = st.mean(x1), st.stdev(x1) / len(x1) ** .5
m2 = st.mean(x2) if x2 else float("nan")
out["R"] = {"baseline_n": len(x1), "baseline_mean": m1, "baseline_se": se1, "threshold": m1 - 2 * se1,
            "two_island_n": len(x2), "two_island_mean": m2, "two_island_se": (st.stdev(x2) / len(x2) ** .5) if len(x2) > 1 else None,
            "per_island": {i: [round(e["reward_mean"], 4) for e in rounds if e.get("island_id") == i] for i in sorted({e.get("island_id") for e in rounds})}}
out["R_pass"] = bool(x2) and m2 >= m1 - 2 * se1
fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
bad = [(e.get("island_id"), e.get("local_round_id")) for e in rounds
       if not (fin(e.get("grad_norm")) and fin(e.get("delta_l2_norm")) and fin(e.get("reward_mean")) and e["grad_norm"] > 0 and e["delta_l2_norm"] > 0)]
hashes = {}
for e in ev:
    if e.get("event") == "rl_policy_apply":
        hashes.setdefault(e["policy_version"], {}).setdefault(e["island_id"], set()).add(e.get("sync/global_policy_hash"))
mismatch = {v: {i: sorted(h) for i, h in d.items()} for v, d in hashes.items()
            if len(set().union(*d.values())) != 1}
both = sorted(v for v, d in hashes.items() if len(d) == 2)
out["H"] = {"rounds": len(rounds), "bad_rounds": bad, "versions_applied_by_both": both, "hash_mismatch": mismatch,
            "grad_norm": [round(e["grad_norm"], 4) for e in rounds], "delta_l2": [round(e["delta_l2_norm"], 5) for e in rounds if fin(e.get("delta_l2_norm"))]}
out["H_pass"] = bool(rounds) and not bad and not mismatch and len(both) > 0
out["pass"] = out["B_pass"] and out["R_pass"] and out["H_pass"]
print(json.dumps(out, indent=1, default=str))
