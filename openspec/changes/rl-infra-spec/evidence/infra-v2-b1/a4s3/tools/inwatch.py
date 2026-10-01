# in-container trigger: python3 inwatch.py '<json list of [phase, rollout_id, request_id, body]>'
import json, os, sys, time
trig = json.loads(sys.argv[1]); tape = os.path.expanduser("~/yeto-output/rl-island-0.jsonl")
inbox = os.path.expanduser("~/yeto-rl/elastic-state/inbox"); log = open(os.path.expanduser("~/yeto-rl/inwatch.log"), "a")
done = set(); pos = 0
while len(done) < len(trig):
    try:
        with open(tape) as f:
            f.seek(pos); data = f.read(); pos = f.tell()
    except FileNotFoundError:
        data = ""
    for line in data.splitlines():
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "rl_driver_phase": continue
        for i, t in enumerate(trig):
            ph, rid, req, body = t[:4]; verb = t[4] if len(t) > 4 else "request"
            if i in done or e.get("phase") != ph or e.get("rollout_id") != rid: continue
            os.makedirs(inbox, exist_ok=True)
            p = os.path.join(inbox, f"{req}.{verb}.json"); tmp = p + ".tmp"
            open(tmp, "w").write(json.dumps(body)); os.replace(tmp, p)
            done.add(i)
            log.write(json.dumps({"submitted": req, "trigger": [ph, rid], "event_time": e.get("time_unix"), "wall": time.time()}) + "\n"); log.flush()
    time.sleep(0.5)
