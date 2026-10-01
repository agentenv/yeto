"""Pre-declared G3 criteria 1-5 (plan.md)."""
import ast, hashlib, json, math, re, sys
from pathlib import Path
d = Path(sys.argv[1]); c = {}
sha = __import__("hashlib").sha256(json.dumps(json.load(open(d.parent / "tis.json")), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
sys.path.insert(0, "/home/michael/work/algo-1a")
from yeto.rl.engine.algorithm import AlgorithmSpec
sha = AlgorithmSpec.from_json_file(str(d.parent / "tis.json")).sha256()
sync = [json.loads(x) for x in (d / "yeto-tape.jsonl").read_text().splitlines() if x.strip()]
steps = [e for e in sync if "sync/responders" in e]
c["1_rc_ok"] = (d / "rc").read_text().strip() == "0"
c["1_three_outer_steps_2_responders_no_stale"] = len(steps) == 3 and all(
    e["sync/responders"] == 2 and e.get("sync/rejected_stale_updates", 0) == 0 for e in steps)
def _load(path):
    out, bad = [], 0
    for x in path.read_text().splitlines():
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            bad += 1  # truncated line from a pull that raced the writer
    return out, bad
loaded = [_load(d / "events" / f"algo1a-g3b-l{i}-modal.jsonl") for i in (0, 1)]
isl = [ev for ev, _ in loaded]
sel = [[e for e in ev if e.get("event") == "rl_engine_selected"] for ev in isl]
c["2_same_sha_no_unverified"] = all(s and s[0].get("rl/algorithm_spec_sha256") == sha and
                                    "rl/unverified_mechanisms" not in s[0] for s in sel)
pubs = [{e["policy_version"]: e["sync/publication_payload_hash"] for e in ev if e.get("event") == "rl_publication"} for ev in isl]
c["3_publication_hashes_equal_v0_3"] = all(v in pubs[0] and v in pubs[1] and pubs[0][v] == pubs[1][v] for v in range(4))
bad = {"rl_invariant_failed", "rl_round_failed", "rl_algorithm_mismatch", "rl_algorithm_island_rejected"}
c["4_rounds_finite_no_failures"] = all(
    len([e for e in ev if e.get("event") == "rl_local_round"]) == 3
    and all(math.isfinite(e["grad_norm"]) for e in ev if e.get("event") == "rl_local_round")
    and not any(e.get("event") in bad for e in ev) for ev in isl)
log = (d / "launch.log").read_text(errors="replace")
per = {}
for m in re.finditer(r"\[algo1a-g3b-(l\d)-modal\].*?log_utils\.py:\d+ - step (\d+): (\{.*\})", log):
    per.setdefault(m.group(1), {})[int(m.group(2))] = ast.literal_eval(m.group(3))
c["5_tis_metrics_each_step"] = sorted(per) == ["l0", "l1"] and all(
    sorted(s) == [0, 1, 2] and all(all(k in v for k in ("train/tis", "train/tis_abs", "train/tis_clipfrac")) for v in s.values())
    for s in per.values())
res = {"unparsable_tape_lines": [b for _, b in loaded], "checks": c, "pass": all(c.values()), "publication_hashes": pubs, "sha": sha,
       "sync_steps": [{k: e[k] for k in ("sync/base_version", "sync/responders") if k in e} for e in steps]}
(d / "check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["checks"]), res["pass"])
