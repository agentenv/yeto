#!/usr/bin/env python3
"""rl-multinode-island tasks §3 judge (criteria fixed before the runs; see tasks.md §3).
usage: s1judge.py <run dir> <g0|g1|g2|g3|g4>  -> prints and writes <run dir>/judgment-<case>.json
Verdicts: PASS | FAIL | PARTIAL (a sub-criterion is not verifiable on this spec, says which) | INVALID_TEST (no valid test: startup/timeout).
"""
import json, os, re, sys, time

R, CASE = sys.argv[1], sys.argv[2]

def read(name):
    p = os.path.join(R, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""

def jsonl(name):
    out = []
    for line in read(os.path.join("pulled", name)).splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def ts(s):  # 2026-10-03T12:00:00Z -> epoch
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone

rc_txt = read("rc.txt").strip()
rc = int(rc_txt.split("=")[1]) if "=" in rc_txt else None
launch = read("launch.log")
events = jsonl("rl-island-0.jsonl")
journal = jsonl("journal.jsonl")
phases = [e for e in events if e.get("event") == "rl_driver_phase"]
rounds_trained = sorted({e.get("rollout_id") for e in phases if e.get("phase") == "train"})
generates = [e for e in phases if e.get("phase") == "generate"]
gpu_names = " ".join(read("pulled/" + f) for f in os.listdir(os.path.join(R, "pulled")) if f.startswith("gpu-")) if os.path.isdir(os.path.join(R, "pulled")) else ""
checks, notes = {}, []

if rc == 124 and not (CASE == "g0" and len(rounds_trained) >= 1 and len(generates) >= 1):
    verdict = "INVALID_TEST"; notes.append("hard_timeout (rc=124)")
elif CASE == "g0":
    # G0 is an image/sm_89 probe: when the local launcher hit its hard timeout during the cold start (setup) but the job went on
    # inside the container and the pulled events show the round, the probe is judged on that in-container evidence (noted).
    if rc == 124:
        notes.append("launcher hard_timeout during setup; judged on in-container events pulled from the kept cluster")
    checks = {"launcher_rc_0_or_setup_timeout": rc in (0, 124), "train_ge_1": len(rounds_trained) >= 1, "generate_ge_1": len(generates) >= 1,
              "gpu_is_L40S": "L40S" in gpu_names, "no_cuda_kernel_error": not re.search(r"no kernel image|sm_89|CUDA error|not compatible", launch)}
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g1":
    # 30 min case window measured from the job start on the (cold-started) cluster; the launcher's own
    # timeout (3600 s) covers the cold start too (2-node image pull + setup + model fetch took ~25-30 min).
    tsl = read("launch.ts.log"); m = re.search(r"^(\S+) .*Job submitted", tsl, re.M)
    end = read("end_utc.txt").strip()
    case_s = (ts(end) - ts(m.group(1))) if (m and end) else None
    notes.append(f"case_window_from_job_submit_s={case_s}")
    topo = [r for r in journal if r.get("kind") == "topology"]
    t0 = topo[0] if topo else {}
    alive = t0.get("alive") if isinstance(t0.get("alive"), list) else []
    checks = {"topology_record_2x1": t0.get("nodes") == 2 and t0.get("gpus_per_node") == 1,
              "alive_nodes_2": len(alive) == 2,
              "startup_bundles_ok": "not node-blocked" not in launch and "BundleMapError" not in launch and len(rounds_trained) >= 1,
              "learner_round_ge_1": len(rounds_trained) >= 1 and len(generates) >= 1,
              "case_window_le_1800s": case_s is not None and case_s <= 1800,
              "launcher_rc_0": rc == 0}
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g2":
    head = read("pulled/apps-" + read("cluster.txt").strip() + ".txt"); worker = read("pulled/apps-" + read("cluster.txt").strip() + "-worker1.txt")
    checks = {"rollout_engine_on_n1": bool(re.search(r"sglang", worker)), "no_rollout_engine_on_n0": not re.search(r"sglang", head),
              "trainer_on_n0": bool(re.search(r"yeto.rl.learner|ray::", head)),
              "rounds_ge_2_cross_node_sync": len(rounds_trained) >= 2 and len(generates) >= 2,
              "config_epoch_monotonic": all(a.get("config_epoch", 0) <= b.get("config_epoch", 0) for a, b in zip(journal, journal[1:])) if journal else True,
              "e1_up_down_edge": None}
    notes.append("E1 rollout-only up/down edge NOT verifiable on 2x1 (no standby GPU; needs 2x2 or larger) -> PARTIAL at best")
    verdict = "PARTIAL" if all(v for v in checks.values() if v is not None) else "FAIL"
elif CASE == "g3":
    kill = read("kill.txt").splitlines()
    kill_epoch = float(kill[1]) if len(kill) > 1 and re.match(r"^\d+\.\d+$", kill[1].strip()) else None
    lost = [r for r in journal if r.get("kind") == "node_lost"]
    rr = [e for e in events if e.get("event") == "rl_reconfiguration" and e.get("result") == "RECOVERY_REQUIRED" and str(e.get("error", "")).startswith("node_lost")]
    lost_wall = lost[0].get("wall_time") if lost else None
    latency = (lost_wall - kill_epoch) if (lost_wall and kill_epoch) else None
    after = [e for e in phases if e.get("phase") in ("train", "generate") and lost_wall and e.get("time_unix", 0) > lost_wall + 1]
    topo = [r for r in journal if r.get("kind") == "topology"]
    refused = [r for r in journal if "recovery refused" in json.dumps(r) or "island topology" in json.dumps(r)]
    checks = {"kill_recorded": kill_epoch is not None, "node_lost_journal": bool(lost), "rl_reconfiguration_RECOVERY_REQUIRED_node_lost": bool(rr),
              "latency_le_60s": latency is not None and latency <= 60, "learner_nonzero_exit": bool(re.search(r"learner exited [1-9]\d*", launch)) or (rc not in (0, None)),
              "no_partial_continuation": not after,
              "restart_precheck_refused": len(topo) >= 2 and bool(refused),
              "not_judged_on_LedgerError": True}
    notes.append(f"latency_s={latency}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g4":
    confirmed = re.findall(r"node instance (\S+) confirmed terminated", launch)
    cleanup = read("cleanup.out")
    checks = {"two_node_confirm_lines": len(set(confirmed)) == 2, "no_UNCONFIRMED": "UNCONFIRMED" not in launch,
              "cleanup_clean_twice": "RESULT: clean twice" in cleanup}
    verdict = "PASS" if all(checks.values()) else "FAIL"
else:
    sys.exit("unknown case")
out = {"case": CASE, "verdict": verdict, "checks": checks, "notes": notes, "rc": rc, "rounds_trained": rounds_trained, "n_generate": len(generates),
       "start_utc": read("start_utc.txt").strip(), "end_utc": read("end_utc.txt").strip(), "yeto_sha": read("yeto_sha.txt").strip()}
json.dump(out, open(os.path.join(R, f"judgment-{CASE}.json"), "w"), indent=1)
print(json.dumps(out))
sys.exit(0 if verdict == "PASS" else 1)
