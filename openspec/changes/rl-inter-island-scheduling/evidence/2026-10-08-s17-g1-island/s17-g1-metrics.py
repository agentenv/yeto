#!/usr/bin/env python3
"""S17 G1 (N3): extra metrics from a saved run (launch.log events, deduped): per-island round times
(rollout/train seconds, round interval), GPU memory/util/power peaks from rl_resource_sample, response length,
first event / first round time since start. usage: metrics.py <run dir> [events.jsonl]"""
import json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1]); src = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "launch.log"
ev, seen = [], set()
for line in src.read_text(errors="replace").splitlines():
    at = line.find("YETO_RL_EVENT "); s = line[at + 14:].strip() if at >= 0 else line.strip()
    if not s.startswith("{") or s in seen: continue
    seen.add(s)
    try: ev.append(json.loads(s))
    except ValueError: pass
out = {}
for isl in sorted({e.get("island_id") for e in ev if e.get("island_id") is not None}):
    E = [e for e in ev if e.get("island_id") == isl]
    rounds = [e for e in E if e.get("event") == "rl_local_round"]
    rs = [e for e in E if e.get("event") == "rl_resource_sample"]
    tr = [e for e in E if e.get("event") == "rl_round_trained"]
    t0 = min(e["time_unix"] for e in E if "time_unix" in e)
    g = [x for e in rs for x in e.get("gpus") or []]
    out[isl] = {
        "rounds": len(rounds),
        "rollout_s": [round(e.get("rollout_seconds") or 0, 1) for e in rounds],
        "train_s": [round(e.get("train_seconds") or 0, 1) for e in rounds],
        "round_interval_s": [round(b["time_unix"] - a["time_unix"], 1) for a, b in zip(rounds, rounds[1:])],
        "first_round_after_first_event_s": round(rounds[0]["time_unix"] - t0, 1) if rounds else None,
        "resp_len_mean": [round(e.get("resp_len_mean") or 0, 1) for e in tr],
        "tok_per_s": [round(e.get("tok_per_s") or 0, 1) for e in tr],
        "gpu_mem_used_mb_max": max((x.get("mem_used_mb") or 0) for x in g) if g else None,
        "gpu_util_pct_max": max((x.get("util_pct") or 0) for x in g) if g else None,
        "gpu_util_pct_mean": round(st.mean(x.get("util_pct") or 0 for x in g), 1) if g else None,
        "gpu_power_w_max": max((x.get("power_w") or 0) for x in g) if g else None,
        "cpu_rss_gb_max": round(max(e.get("cpu_rss_bytes") or 0 for e in rs) / 2**30, 2) if rs else None,
        "resource_samples": len(rs),
        "publish_apply_s": [round(e.get("sync/apply_seconds") or 0, 2) for e in E if e.get("event") == "rl_policy_apply"],
    }
print(json.dumps(out, indent=1))
