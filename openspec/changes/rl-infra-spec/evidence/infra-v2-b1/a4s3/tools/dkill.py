# in-container fault injector for E1-D (1)(2): python3 dkill.py '<json rules>'
# rule: {"name":..., "tx":"<request id suffix of tx_id>", "when":{"kind":"fork_op","op":"start","status":"issued"} | {"kind":"phase","phase":"VERIFYING"},
#        "gpu": 6, "mode": "when_proc_appears" | "immediate", "min_age_s": 10}
# Reads ~/yeto-rl/elastic-state/reconfig/journal.jsonl, kills (SIGKILL) the compute processes on the given GPU index. Log: ~/yeto-rl/dkill.log
import json, os, signal, subprocess, sys, time
rules = json.loads(sys.argv[1]); J = os.path.expanduser("~/yeto-rl/elastic-state/reconfig/journal.jsonl")
log = open(os.path.expanduser("~/yeto-rl/dkill.log"), "a")
def L(**k):
    k["wall"] = time.time(); log.write(json.dumps(k) + "\n"); log.flush()
def sh(c):
    return subprocess.run(c, capture_output=True, text=True).stdout
def uuids():
    return {int(a.split(",")[0]): a.split(",")[1].strip() for a in sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines() if a.strip()}
UU = uuids()
def procs(gpu):
    out = set()
    for l in sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]).splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) >= 2 and p[0] == UU.get(gpu): out.add(int(p[1]))
    return out
def kill(gpu, cache):
    res = {}
    for rnd in (0, 1):
        for p in sorted(set(cache) if rnd == 0 else procs(gpu)):
            try: os.kill(p, signal.SIGKILL); res[p] = "killed"
            except Exception as e: res.setdefault(p, repr(e)[:80])
    return res
pos = 0; state = {r["name"]: {"armed": False, "done": False} for r in rules}; cache = {}; first = {}; last_s = 0.0; cur = {}
L(event="start", uuids=UU)
while not all(s["done"] for s in state.values()):
    try:
        with open(J) as f: f.seek(pos); data = f.read(); pos = f.tell()
    except FileNotFoundError: data = ""
    for line in data.splitlines():
        try: e = json.loads(line)
        except Exception: continue
        for r in rules:
            s = state[r["name"]]
            if s["armed"] or s["done"] or not str(e.get("tx_id", "")).endswith("-" + r["tx"]): continue
            if all(e.get(k) == v for k, v in r["when"].items()):
                s["armed"] = True; s["t_armed"] = time.time(); L(event="armed", rule=r["name"], journal=e.get("seq"))
                if r["mode"] == "immediate":
                    s["done"] = True; L(event="kill", rule=r["name"], gpu=r["gpu"], res=kill(r["gpu"], cache.get(r["gpu"], ())))
    now = time.time()
    if now - last_s >= 1.0:
        last_s = now
        for r in rules:
            ps = procs(r["gpu"])
            if ps: cache[r["gpu"]] = ps
            cur[r["gpu"]] = ps
    for r in rules:
        g = r["gpu"]; ps = cur.get(g, set())
        s = state[r["name"]]
        if s["armed"] and not s["done"] and r["mode"] == "when_proc_appears":
            if ps:
                first.setdefault(r["name"], time.time())
                if time.time() - first[r["name"]] >= r.get("min_age_s", 10):
                    s["done"] = True; L(event="kill", rule=r["name"], gpu=g, res=kill(g, ps))
            else: first.pop(r["name"], None)
    time.sleep(0.05)
