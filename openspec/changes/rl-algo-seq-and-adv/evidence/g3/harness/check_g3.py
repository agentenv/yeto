"""Pre-declared 7.6 criteria (plan.md). usage: check_g3.py <run dir>"""
import json, math, re, sys
from pathlib import Path
import os
d = Path(sys.argv[1]); c = {}
# checkout of YETO_SHA: argv[2] or $G3_TREE (the runs used /tmp/a2-g3)
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else os.environ.get("G3_TREE", "/tmp/a2-g3"))
from yeto.rl.engine.algorithm import AlgorithmSpec
spec_path = d / "maxrl.json" if (d / "maxrl.json").exists() else d.parent / "maxrl.json"
sha = AlgorithmSpec.from_json_file(str(spec_path)).sha256()
rc = (d / "rc").read_text().strip()
c["0_exit_code"] = rc  # 0 pass; 2 -> tapes decide; 3 or other -> fail
sync = [json.loads(x) for x in (d / "yeto-tape.jsonl").read_text().splitlines() if x.strip()] \
    if (d / "yeto-tape.jsonl").exists() else []
steps = [e for e in sync if "sync/responders" in e]
c["1_three_outer_steps_2_responders_no_stale"] = len(steps) == 3 and all(
    e["sync/responders"] == 2 and e.get("sync/rejected_stale_updates", 0) == 0 for e in steps)
files = sorted((d / "events").glob("*.jsonl"))
isl, bad_lines = [], []
for f in files:
    ev, bad = [], 0
    for x in f.read_text().splitlines():
        try:
            ev.append(json.loads(x))
        except json.JSONDecodeError:
            bad += 1
    isl.append(ev); bad_lines.append(bad)
c["tapes"] = [f.name for f in files]
two = len(isl) == 2
sel = [[e for e in ev if e.get("event") == "rl_engine_selected"] for ev in isl]
c["2_same_sha_no_unverified"] = two and all(
    s and s[0].get("rl/algorithm_spec_sha256") == sha and not s[0].get("rl/unverified_mechanisms")
    for s in sel)
pubs = [{e["policy_version"]: e.get("sync/publication_payload_hash") for e in ev
         if e.get("event") == "rl_publication"} for ev in isl]
c["3_publication_hashes_equal_v0_3"] = two and all(
    v in pubs[0] and v in pubs[1] and pubs[0][v] is not None and pubs[0][v] == pubs[1][v] for v in range(4))
failing = {"rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch", "rl_algorithm_island_rejected"}
def rounds(ev):
    return [e for e in ev if e.get("event") in ("rl_local_round", "rl_round_trained")]
c["4_rounds_finite_finalized_no_failures"] = two and all(
    len([e for e in ev if e.get("event") == "rl_local_round"]) == 3
    and len([e for e in ev if e.get("event") == "rl_round_trained"]) == 3
    and all(math.isfinite(float(e["grad_norm"])) for e in rounds(ev) if "grad_norm" in e)
    and any(e.get("event") == "rl_learner_finalized" for e in ev)
    and not any(e.get("event") in failing for e in ev) for ev in isl)
log = (d / "launch.log").read_text(errors="replace")
obs = {"transform_events_per_island": {m: len(re.findall(rf"\[algo2a-g3-{m}-modal\].*rl_advantage_transform", log)) for m in ("l0", "l1")},
       "nonzero_advantages_per_round": [[(e.get("rollout_id"), e.get("nonzero_advantages")) for e in ev if e.get("event") == "rl_round_trained"] for ev in isl],
       "grad_norms": [[(e.get("rollout_id"), e.get("grad_norm")) for e in rounds(ev)] for ev in isl]}
ok = c["0_exit_code"] in ("0", "2") and all(v for k, v in c.items() if k[0] in "1234")
res = {"checks": c, "pass": ok, "sha": sha, "publication_hashes": pubs, "unparsable_lines": bad_lines,
       "sync_steps": [{k: e[k] for k in ("sync/base_version", "sync/responders") if k in e} for e in steps],
       "observations": obs}
(d / "check.json").write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(c), ok)
