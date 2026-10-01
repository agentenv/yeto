#!/bin/bash
# Independent reclaim: after $1 seconds terminate sandbox $2 and stop the app (setsid, survives agent).
sleep "$1"; /tmp/modal-venv/bin/python "$(dirname "$0")/sbx.py" kill "$2" >/dev/null 2>&1
/tmp/modal-venv/bin/modal app stop -y algo1a-g1 >/dev/null 2>&1; echo "watchdog fired $(date -u +%FT%TZ)"
