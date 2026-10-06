#!/usr/bin/env python3
"""In-container E1 trigger for the m3 case (same mechanism as chain 8's inwatch.py, gpu-b1-runs):
usage: python3 s1inwatch.py '<json list of [phase, rollout_id, request_id, body]>'
Tails the island tape (~/yeto-output/rl-island-0.jsonl); when an rl_driver_phase event with the
given phase/rollout_id appears, drops <request_id>.request.json (the plan body: target config,
expected_config_epoch, deadline_s) into the elastic inbox (~/yeto-rl/elastic-state/inbox), which
the IslandController polls (`yeto rl` reconfigure request). Log: ~/yeto-rl/inwatch.log."""
import json, os, sys, time

trig = json.loads(sys.argv[1])
tape = os.path.expanduser("~/yeto-output/rl-island-0.jsonl")
inbox = os.path.expanduser("~/yeto-rl/elastic-state/inbox")
log = open(os.path.expanduser("~/yeto-rl/inwatch.log"), "a")
done, pos = set(), 0
while len(done) < len(trig):
    try:
        with open(tape) as f:
            f.seek(pos); data = f.read(); pos = f.tell()
    except FileNotFoundError:
        data = ""
    for line in data.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "rl_driver_phase":
            continue
        for i, t in enumerate(trig):
            ph, rid, req, body = t[:4]
            verb = t[4] if len(t) > 4 else "request"
            if i in done or e.get("phase") != ph or e.get("rollout_id") != rid:
                continue
            os.makedirs(inbox, exist_ok=True)
            p = os.path.join(inbox, f"{req}.{verb}.json"); tmp = p + ".tmp"
            open(tmp, "w").write(json.dumps(body)); os.replace(tmp, p)
            done.add(i)
            log.write(json.dumps({"submitted": req, "trigger": [ph, rid], "event_time": e.get("time_unix"),
                                  "wall": time.time()}) + "\n"); log.flush()
    time.sleep(0.5)
