#!/bin/bash
sleep 1080
/tmp/modal-venv/bin/modal app list --json | /tmp/modal-venv/bin/python -c '
import json,sys,subprocess
for a in json.load(sys.stdin):
    if a["description"].startswith("algo2b-t4-") and a["state"] not in ("stopped",):
        print("watchdog stop", a["app_id"]); subprocess.run(["/tmp/modal-venv/bin/modal","app","stop","-y",a["app_id"]])
'
