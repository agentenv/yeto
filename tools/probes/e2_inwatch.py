"""In-container trigger for the E2 C3 runs (installed by run.sh via `modal container exec`).

usage: python3 e2_inwatch.py <phase> <rollout_id> <request_id> <expected_epoch> <deadline_s>

Tails the learner tape; when ``rl_driver_phase`` with (phase, rollout_id) appears, writes the
controller command file ``<state>/inbox/<request_id>.rebuild.json`` (same body as
``python -m yeto.rl.engine.controller rebuild-trainer``). Triggering during round 2's
``train`` puts the rebuild at the safe point before round 3 (cut local_step = 3).
"""
import json
import os
import sys
import time

phase, rid, req, epoch, deadline = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
tape = os.path.expanduser("~/yeto-output/rl-island-0.jsonl")
inbox = os.path.expanduser("~/yeto-rl/elastic-state/inbox")
log = open(os.path.expanduser("~/yeto-rl/inwatch.log"), "a")
pos = 0
while True:
    try:
        with open(tape) as fh:
            fh.seek(pos)
            data = fh.read()
            pos = fh.tell()
    except FileNotFoundError:
        data = ""
    for line in data.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "rl_driver_phase" and event.get("phase") == phase and event.get("rollout_id") == rid:
            os.makedirs(inbox, exist_ok=True)
            body = {"kind": "trainer-rebuild", "expected_config_epoch": epoch, "deadline_s": deadline}
            path = os.path.join(inbox, f"{req}.rebuild.json")
            with open(path + ".tmp", "w") as fh:
                fh.write(json.dumps(body))
            os.replace(path + ".tmp", path)
            log.write(json.dumps({"submitted": req, "trigger": [phase, rid], "wall": time.time()}) + "\n")
            log.flush()
            sys.exit(0)
    time.sleep(0.5)
