"""S19: offline f / tool-wait share from existing agentic tapes (no GPU).
f_upper  = union(rollout generate compute spans) / run wall  (what recommend.py SERIAL uses)
f_sat    = f_upper * share of in-generate load samples with queued_requests>0
           (engine had a backlog => more engines could have served it sooner)
tool_wait_share = mean over in-generate samples of tool_wait_trajectories / harness_in_flight
"""
import json, sys, glob, statistics as st
sys.path.insert(0, "/home/michael/work/s19-elastic")
from yeto.rl.engine.timeline import load_windows, _union

RUNS = ["s17-n17-ab-a-20261009a", "s17-n17-ab-a-20261009b", "s17-n17-ab-b-20261009a",
        "s17-n17-ab-b-20261009b", "s18-aru3-mb-20261009b", "s18-aru3-m0-20261009a",
        "s18-aru3-m1-20261009a"]
out = {}
for r in RUNS:
    path = glob.glob(f"/home/michael/work/s1-runs/{r}/runs/*/events/*-l0-modal.jsonl")[0]
    ev = [json.loads(l) for l in open(path) if l.strip()]
    sp = [e for e in ev if e.get("event") == "rl_timeline_span"]
    gen = [(e["start"], e["end"]) for e in sp if e["kind"] == "compute"
           and "rollout" in e["role"].split("+") and e.get("task") != "eval"]
    t0 = min(e["start"] for e in sp); t1 = max(e["end"] for e in sp)
    wall = t1 - t0
    g = _union(gen)
    ls = [e for e in ev if e.get("event") == "rl_load_sample" and e.get("t") is not None
          and any(s <= e["t"] <= en for s, en in gen)]
    q = sum(1 for e in ls if e["queued_requests"] > 0) / len(ls)
    tw = [e["tool_wait_trajectories"] / e["harness_in_flight"] for e in ls if e["harness_in_flight"]]
    idle_tool = sum(1 for e in ls if e["running_requests"] == 0 and e["tool_wait_trajectories"] > 0) / len(ls)
    kv = max(e["kv_used_tokens"] / e["kv_capacity_tokens"] for e in ls if e.get("kv_capacity_tokens"))
    rounds = sum(1 for e in ev if e.get("event") == "rl_round_trained")
    ws = load_windows(ev, 60.0)
    mean = lambda k: st.mean([getattr(w, k) for w in ws if getattr(w, k) is not None])
    out[r] = dict(events=path, rounds=rounds, wall_s=round(wall, 1),
                  wall_per_round_s=round(wall / rounds, 1), generate_s=round(g, 1),
                  f_upper=round(g / wall, 3), queued_share_in_generate=round(q, 3),
                  f_sat=round(g / wall * q, 3), tool_wait_share=round(st.mean(tw), 3),
                  engine_idle_tool_wait_share=round(idle_tool, 3), kv_peak=round(kv, 3),
                  n_samples_in_generate=len(ls),
                  timeline_60s=dict(n=len(ws), rollout_busy=round(mean("rollout_busy_fraction"), 3),
                                    tool_wait=round(mean("tool_wait_fraction"), 3),
                                    tail_wait=round(mean("tail_wait_fraction"), 3)))
json.dump(out, open("/home/michael/work/s1-runs/s19-elastic-f/result.json", "w"), indent=1)
for k, v in out.items():
    print(k, {x: y for x, y in v.items() if x != "events"})
