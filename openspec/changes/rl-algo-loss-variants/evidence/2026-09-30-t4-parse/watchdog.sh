#!/bin/bash
# Stops only algo2b-t4-* apps created after this watchdog started, 1080 s later.
START=$(date -u +"%Y-%m-%d %H:%M:%S")
sleep 1080
/tmp/modal-venv/bin/modal app list --json | START="$START" /tmp/modal-venv/bin/python -c '
import json,os,sys,subprocess
s=os.environ["START"]
for a in json.load(sys.stdin):
    if a["description"].startswith("algo2b-t4-") and a["state"]!="stopped" and a["created_at"][:19]>=s:
        print("watchdog stop", a["app_id"]); subprocess.run(["/tmp/modal-venv/bin/modal","app","stop","-y",a["app_id"]])
'
