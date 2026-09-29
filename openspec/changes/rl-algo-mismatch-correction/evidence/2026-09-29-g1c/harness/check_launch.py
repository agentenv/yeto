"""check_launch.py <run> <rundir> <allow...>: criteria 1-6 of plan.md."""
import json, subprocess, sys, os
from pathlib import Path
run, d, allow = sys.argv[1], Path(sys.argv[2]), sorted(sys.argv[3:])
sys.path.insert(0, "/home/michael/work/algo-1a")
from yeto.rl.engine.algorithm import AlgorithmSpec
spec = AlgorithmSpec.from_json_file(str(d / "spec.json"))
(d / "expected.json").write_text(json.dumps({"sha256": spec.sha256(), "allow": allow}))
(d / "island-0").mkdir(exist_ok=True)
if (d / "tape.jsonl").exists():
    (d / "island-0" / "events.jsonl").write_text((d / "tape.jsonl").read_text())
if not (d / "miles.log").exists():
    os.symlink("launch.log", d / "miles.log")
p = subprocess.run([sys.executable, str(Path(__file__).parent / "check_g1.py"), run, str(d), "3", "no-sync"],
                   capture_output=True, text=True)
res = json.loads(p.stdout)
rc = (d / "rc").read_text().strip()
log = (d / "launch.log").read_text(errors="replace")
res["checks"]["rc0"] = rc == "0" or (rc == "2" and "is not fetchable over ssh" in log)
ev = [json.loads(x) for x in (d / "island-0" / "events.jsonl").read_text().splitlines()] \
    if (d / "island-0" / "events.jsonl").exists() else []
rounds = [e for e in ev if e.get("event") == "rl_local_round"]
pubs = [e for e in ev if e.get("event") == "rl_publication"]
res["checks"]["receipt_round_stats"] = (
    [e.get("local_round_id") for e in rounds] == [1, 2, 3]
    and [e.get("base_policy_version") for e in rounds] == [0, 1, 2]
    and all(e.get("completed_trajectories") == 32 and (e.get("action_tokens") or 0) > 0 for e in rounds)
    and [e.get("policy_version") for e in pubs] == [0, 1, 2, 3])
res["pass"] = all(res["checks"].values())
(d / "check.json").write_text(json.dumps(res, indent=1, default=str))
print(run, "PASS" if res["pass"] else "FAIL", [k for k, v in res["checks"].items() if not v])
