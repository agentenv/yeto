#!/bin/bash
# Start one E6 Modal driver under GPU-LAUNCH.lock and release the lock once the
# detached app exists (app_id.txt) plus 60 s; the driver keeps running unlocked.
# usage: launch_locked.sh OUT_DIR LOG -- driver argv...
set -u
OUT=$1; LOG=$2; shift 3
LOCK=/home/michael/work/infra-drafts/GPU-LAUNCH.lock
exec 9>"$LOCK"
flock 9
T=$(ps -u michael -L --no-headers | wc -l)
while [ "$T" -ge 2950 ]; do echo "waiting: $T threads" >> "$LOG"; sleep 30; T=$(ps -u michael -L --no-headers | wc -l); done
mkdir -p "$OUT"
"$@" 9>&- >> "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 120); do [ -s "$OUT/app_id.txt" ] && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
sleep 60
flock -u 9; exec 9>&-
wait $PID
