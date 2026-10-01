# in-container helper for E1-D (5)/(7): python3 dctl.py marker | kill_then_up '<json request body>' 
import json, os, signal, subprocess, sys, time
mode = sys.argv[1]; H = os.path.expanduser("~/yeto-rl"); SD = H + "/elastic-state"; TAPE = os.path.expanduser("~/yeto-output/rl-island-0.jsonl")
log = open(H + "/dctl.log", "a")
def L(**k):
    k["wall"] = time.time(); log.write(json.dumps(k) + "\n"); log.flush()
def tape():
    try: return [json.loads(l) for l in open(TAPE).read().splitlines() if l.strip().startswith("{")]
    except Exception: return []
def journal():
    try: return [json.loads(l) for l in open(SD + "/reconfig/journal.jsonl").read().splitlines() if l.strip()]
    except Exception: return []
def submit(rid, body):
    os.makedirs(SD + "/inbox", exist_ok=True); p = f"{SD}/inbox/{rid}.request.json"
    open(p + ".tmp", "w").write(json.dumps(body)); os.replace(p + ".tmp", p); L(event="submitted", rid=rid)
if mode == "marker":  # (5): suppress the once-per-state-dir kill for the first (up) transaction, re-arm it once that up succeeded
    m = SD + "/test-killed-at-COMMITTED"
    while not os.path.isdir(SD): time.sleep(0.2)
    open(m, "w").write("suppressed for the first transaction (dctl)\n"); L(event="marker_created")
    while not any(r.get("kind") == "phase" and r.get("phase") == "SUCCEEDED" and r.get("request_id") == sys.argv[2] for r in journal()): time.sleep(0.2)
    os.remove(m); L(event="marker_removed")
if mode == "kill_then_up":  # (7): SIGKILL the learner after generate(rollout_id 1), let the restart loop restart it, then submit the up
    body = json.loads(sys.argv[3]); rid = sys.argv[2]
    while not any(e.get("event") == "rl_driver_phase" and e.get("phase") == "generate" and e.get("rollout_id") == 1 for e in tape()): time.sleep(0.3)
    pids = [p for p in subprocess.run(["pgrep", "-f", "[y]eto.rl.learner"], capture_output=True, text=True).stdout.split()
            if "python" in (os.path.realpath(f"/proc/{p}/exe") if os.path.exists(f"/proc/{p}/exe") else "")]
    L(event="learner_pids", pids=pids)
    for p in pids: os.kill(int(p), signal.SIGKILL)
    L(event="killed", pids=pids)
    n0 = sum(1 for e in tape() if e.get("event") == "rl_driver_start")
    while sum(1 for e in tape() if e.get("event") == "rl_driver_start") < 2: time.sleep(0.3)
    L(event="second_driver_start")
    seen = len(tape())
    while True:
        ev = tape()
        if any(e.get("event") == "rl_driver_phase" and e.get("phase") == "train" for e in ev[seen:]): break
        time.sleep(0.3)
    submit(rid, body)
