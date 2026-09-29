"""2.4 per plan.md: median round wall (rounds 2-6) from rl_round_trained deltas; GPU-hours = 4 x run wall."""
import datetime as dt, json, statistics, sys, pathlib
root = pathlib.Path(sys.argv[1])
rows = {}
for d in sorted(root.glob("*-s*")):
    tape = [json.loads(l) for l in open(d / "tape.jsonl")]
    t = [e["time_unix"] for e in tape if e["event"] == "rl_round_trained"]
    deltas = [b - a for a, b in zip(t, t[1:])]  # round k wall for k=2..N
    phases = {}
    ev = [e for e in tape if e["event"] == "rl_driver_phase"]
    for a, b in zip(ev, ev[1:]):
        phases[a["phase"]] = phases.get(a["phase"], 0.0) + b["time_unix"] - a["time_unix"]
    start = dt.datetime.fromisoformat(open(d / "start_utc.txt").read().strip().replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(open(d / "end_utc.txt").read().strip().replace("Z", "+00:00"))
    rows[d.name] = {
        "rounds": len(t), "round_wall_s": deltas,
        "median_round_s": statistics.median(deltas) if deltas else None,
        "phase_s": {k: round(v, 1) for k, v in phases.items()},
        "run_wall_h": (end - start).total_seconds() / 3600,
        "pool_gpu_h": 4 * (end - start).total_seconds() / 3600,
        "mode": next((e.get("execution_mode") for e in tape if e["event"] == "rl_driver_start"), None),
    }
def med(cfg, seed): return rows.get(f"{cfg}-s{seed}", {}).get("median_round_s")
verdict = {}
for seed in (17, 29):
    a, b = med("t2r2", seed), med("t1r3", seed)
    verdict[seed] = None if a is None or b is None else ("t2r2" if a <= 0.9 * b else "t1r3" if b <= 0.9 * a else "none")
faster = set(verdict.values())
print(json.dumps({"runs": rows, "per_seed_faster": verdict,
                  "same_profile_faster": faster.pop() if len(faster) == 1 and None not in verdict.values() and "none" not in verdict.values() else "no significant difference"},
                 indent=1, sort_keys=True))
